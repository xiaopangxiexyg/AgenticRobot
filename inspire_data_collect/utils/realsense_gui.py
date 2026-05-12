from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .camera_preview import CameraPreviewConfig, CameraPreviewer
    from .realsense_rgb import (
        RealSenseRGBCameraConfig,
        RealSenseRGBSource,
        load_realsense_camera_config_map,
    )
except ImportError:
    # Allow running this file directly:
    #   python inspire_data_collect/utils/realsense_gui.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from inspire_data_collect.utils.camera_preview import (  # type: ignore[no-redef]
        CameraPreviewConfig,
        CameraPreviewer,
    )
    from inspire_data_collect.utils.realsense_rgb import (  # type: ignore[no-redef]
        RealSenseRGBCameraConfig,
        RealSenseRGBSource,
        load_realsense_camera_config_map,
    )


DEFAULT_CAMERA_CONFIG_PATH = Path(__file__).resolve().parents[1] / "realsense.json"
DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parents[2] / "datasets" / "2026-04-27_chips_single"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open a simple RealSense RGB viewer using the same frame source and "
            "color-space conversion as inspire_data_collect recording."
        )
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG_PATH,
        help="RealSense JSON config to load when --camera-serial is not set.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=["single", "multi"],
        default="single",
        help="'single' expects one configured camera; 'multi' opens every camera in the config.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="",
        help="Camera key for --camera-serial, or the camera to show from --camera-config.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show every configured camera. Equivalent to --camera-mode multi for display.",
    )
    parser.add_argument(
        "--camera-serial",
        type=str,
        default="",
        help="Open one camera directly by serial instead of loading --camera-config.",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-keep-last", type=int, default=4)
    parser.add_argument("--camera-color-space", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--camera-wait-timeout-ms", type=int, default=1000)
    parser.add_argument("--camera-stale-after-sec", type=float, default=2.0)
    parser.add_argument("--camera-auto-exposure-priority", type=int, default=None)
    parser.add_argument(
        "--window-prefix",
        type=str,
        default="realsense_gui",
        help="Prefix for OpenCV preview window names.",
    )
    parser.add_argument(
        "--autosize",
        action="store_true",
        help="Use fixed-size OpenCV windows instead of resizable windows.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip the OpenCV HighGUI display preflight check.",
    )
    parser.add_argument(
        "--status-sec",
        type=float,
        default=2.0,
        help="Seconds between terminal status prints. Set <=0 to disable.",
    )
    parser.add_argument(
        "--view-mode",
        choices=["live", "overlay"],
        default="live",
        help=(
            "'live' shows the RealSense preview exactly like collection time. "
            "'overlay' shows dataset frame and blended comparison in one window."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root used by --view-mode overlay.",
    )
    parser.add_argument(
        "--dataset-camera-key",
        type=str,
        default="front",
        help="Dataset camera key to load from videos/observation.images.<key>.",
    )
    parser.add_argument(
        "--dataset-episode",
        type=int,
        default=0,
        help="Dataset episode index to compare against in --view-mode overlay.",
    )
    parser.add_argument(
        "--dataset-frame-index",
        type=int,
        default=0,
        help="Video frame index within the selected dataset episode.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Live image opacity in overlay mode. 0.5 means half-transparent live image.",
    )
    return parser


def build_camera_sources(args: argparse.Namespace) -> dict[str, RealSenseRGBSource]:
    if args.camera_serial:
        config = RealSenseRGBCameraConfig(
            serial=str(args.camera_serial),
            width=int(args.camera_width),
            height=int(args.camera_height),
            fps=int(args.camera_fps),
            camera_key=str(args.camera_key or "front"),
            keep_last=int(args.camera_keep_last),
            color_space=str(args.camera_color_space),
            wait_timeout_ms=int(args.camera_wait_timeout_ms),
            stale_after_sec=float(args.camera_stale_after_sec),
            auto_exposure_priority=args.camera_auto_exposure_priority,
            metadata={"source": "realsense_gui_cli"},
        )
        return {config.camera_key: RealSenseRGBSource(config)}

    camera_configs = load_realsense_camera_config_map(Path(args.camera_config))
    if not camera_configs:
        raise ValueError(f"No cameras found in config: {args.camera_config}")

    if bool(args.show_all) or str(args.camera_mode) == "multi":
        return {
            camera_key: RealSenseRGBSource(config)
            for camera_key, config in camera_configs.items()
        }

    key = str(args.camera_key or "").strip()
    if key:
        if key not in camera_configs:
            available = ", ".join(sorted(camera_configs))
            raise ValueError(
                f"--camera-key {key!r} not found in {args.camera_config}. "
                f"Available camera keys: {available}"
            )
        config = camera_configs[key]
        return {key: RealSenseRGBSource(config)}

    if len(camera_configs) != 1:
        available = ", ".join(sorted(camera_configs))
        raise ValueError(
            "Multiple cameras are configured. Use --camera-key or --show-all. "
            f"Available camera keys: {available}"
        )
    key, config = next(iter(camera_configs.items()))
    return {key: RealSenseRGBSource(config)}


def format_status(camera_sources: dict[str, RealSenseRGBSource]) -> str:
    chunks: list[str] = []
    for camera_key, source in sorted(camera_sources.items()):
        info = source.device_info()
        frame_count = int(info.get("frame_count", 0))
        last_age = info.get("last_frame_age_sec")
        age_text = "none" if last_age is None else f"{float(last_age):.3f}s"
        error = info.get("last_error")
        if error:
            chunks.append(f"{camera_key}=frames:{frame_count},age:{age_text},error:{error}")
        else:
            chunks.append(f"{camera_key}=frames:{frame_count},age:{age_text}")
    return " | ".join(chunks)


def episode_id_from_index(index: int) -> str:
    return f"episode_{int(index):06d}"


def chunk_id_from_episode_index(index: int, *, chunk_size: int = 1000) -> str:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    return f"chunk-{int(index) // int(chunk_size):03d}"


def dataset_video_path(
    dataset_root: Path,
    *,
    camera_key: str,
    episode_index: int,
) -> Path:
    return (
        Path(dataset_root)
        / "videos"
        / f"observation.images.{str(camera_key)}"
        / chunk_id_from_episode_index(int(episode_index))
        / f"{episode_id_from_index(int(episode_index))}.mp4"
    )


def load_dataset_frame_bgr(
    dataset_root: Path,
    *,
    camera_key: str,
    episode_index: int,
    frame_index: int,
) -> object:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV (cv2) is required for dataset overlay mode.") from exc

    path = dataset_video_path(
        dataset_root,
        camera_key=str(camera_key),
        episode_index=int(episode_index),
    )
    if not path.exists():
        raise FileNotFoundError(f"Dataset video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open dataset video: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0 and int(frame_index) >= frame_count:
            raise ValueError(
                f"--dataset-frame-index {frame_index} is outside {path.name} "
                f"(frame_count={frame_count})."
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame_bgr = cap.read()
    finally:
        cap.release()

    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to decode frame {frame_index} from {path}")
    # Dataset videos are written as BGR frames by utils/writer.py::_coerce_video_frame.
    # OpenCV decodes them back as BGR, which is the exact display-space image.
    return frame_bgr


def blend_live_over_dataset_bgr(
    live_bgr: object,
    dataset_bgr: object,
    *,
    alpha: float,
) -> object:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV (cv2) is required for overlay mode.") from exc

    live = np.asarray(live_bgr).astype(np.uint8, copy=False)
    reference = np.asarray(dataset_bgr).astype(np.uint8, copy=False)
    if live.shape[:2] != reference.shape[:2]:
        raise ValueError(
            "Live preview shape does not match dataset frame shape: "
            f"live={tuple(live.shape)} dataset={tuple(reference.shape)}. "
            "Use matching camera width/height and dataset camera."
        )

    opacity = max(0.0, min(1.0, float(alpha)))
    return cv2.addWeighted(live, opacity, reference, 1.0 - opacity, 0.0)


def draw_panel_label(image: object, text: str) -> object:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV (cv2) is required for overlay mode.") from exc

    output = np.asarray(image).copy()
    cv2.rectangle(output, (12, 12), (248, 48), (20, 20, 20), thickness=-1)
    cv2.putText(
        output,
        str(text),
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return output


def make_overlay_canvas(
    dataset_bgr: object,
    blended_bgr: object,
) -> object:
    reference_panel = draw_panel_label(dataset_bgr, "dataset")
    overlay_panel = draw_panel_label(blended_bgr, "overlay")
    return np.concatenate([reference_panel, overlay_panel], axis=1)


def update_overlay_view(
    *,
    source: RealSenseRGBSource,
    camera_key: str,
    previewer: CameraPreviewer,
    dataset_frame_bgr: object,
    alpha: float,
) -> int | None:
    if not previewer.ensure_available():
        return None

    cv2 = previewer._require_cv2()
    window_name = f"{previewer.config.window_prefix}:overlay:{camera_key}"
    previewer._ensure_window(window_name)

    frame = source.get_latest_frame()
    if frame is None:
        placeholder = previewer._placeholder_image(camera_key=camera_key, source=source)
        if np.asarray(placeholder).shape[:2] != np.asarray(dataset_frame_bgr).shape[:2]:
            raise ValueError(
                "Live placeholder shape does not match dataset frame shape: "
                f"live={tuple(np.asarray(placeholder).shape)} "
                f"dataset={tuple(np.asarray(dataset_frame_bgr).shape)}."
            )
        image_bgr = make_overlay_canvas(dataset_frame_bgr, placeholder)
    else:
        live_bgr = previewer._display_image(frame)
        blended_bgr = blend_live_over_dataset_bgr(
            live_bgr,
            dataset_frame_bgr,
            alpha=float(alpha),
        )
        image_bgr = make_overlay_canvas(dataset_frame_bgr, blended_bgr)
    cv2.imshow(window_name, image_bgr)
    return int(cv2.waitKey(1))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    camera_sources = build_camera_sources(args)
    previewer = CameraPreviewer(
        CameraPreviewConfig(
            window_prefix=str(args.window_prefix),
            preflight_highgui=not bool(args.no_preflight),
            resizable_window=not bool(args.autosize),
        )
    )
    stop_requested = False

    def _request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        for camera_key, source in camera_sources.items():
            print(f"starting RealSense camera '{camera_key}' ({source.config.serial})")
            source.start()

        if not previewer.ensure_available():
            reason = previewer.disabled_reason or "unknown"
            raise RuntimeError(f"OpenCV preview is unavailable: {reason}")

        overlay_source_key = None
        dataset_frame_bgr = None
        if str(args.view_mode) == "overlay":
            if len(camera_sources) != 1:
                available = ", ".join(sorted(camera_sources))
                raise ValueError(
                    "Overlay mode expects exactly one live camera. "
                    f"Select one with --camera-key. Available: {available}"
                )
            overlay_source_key = next(iter(camera_sources))
            dataset_frame_bgr = load_dataset_frame_bgr(
                Path(args.dataset_root),
                camera_key=str(args.dataset_camera_key),
                episode_index=int(args.dataset_episode),
                frame_index=int(args.dataset_frame_index),
            )
            source = camera_sources[overlay_source_key]
            expected_shape = (int(source.config.height), int(source.config.width))
            actual_shape = tuple(np.asarray(dataset_frame_bgr).shape[:2])
            if actual_shape != expected_shape:
                raise ValueError(
                    "Dataset frame shape does not match live camera config: "
                    f"dataset={actual_shape} live={expected_shape}. "
                    "Use matching RealSense width/height/fps config for exact comparison."
                )
            print(
                "overlay reference: "
                f"{dataset_video_path(Path(args.dataset_root), camera_key=str(args.dataset_camera_key), episode_index=int(args.dataset_episode))} "
                f"frame={int(args.dataset_frame_index)} live_alpha={float(args.overlay_alpha):.2f}"
            )

        print("viewer running; press q/esc in the OpenCV window or Ctrl-C to exit")
        last_status_at = 0.0
        while not stop_requested:
            if str(args.view_mode) == "overlay":
                assert overlay_source_key is not None
                assert dataset_frame_bgr is not None
                key = update_overlay_view(
                    source=camera_sources[overlay_source_key],
                    camera_key=str(overlay_source_key),
                    previewer=previewer,
                    dataset_frame_bgr=dataset_frame_bgr,
                    alpha=float(args.overlay_alpha),
                )
            else:
                key = previewer.update(camera_sources)
            key_code = -1 if key is None else int(key) & 0xFF
            if key_code in (ord("q"), 27):
                break
            now = time.time()
            if float(args.status_sec) > 0.0 and now - last_status_at >= float(args.status_sec):
                print(format_status(camera_sources))
                last_status_at = now
            time.sleep(0.001)
    finally:
        previewer.close()
        for source in camera_sources.values():
            source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
