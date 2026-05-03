#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


PORTABLE_ROOT = Path(__file__).resolve().parent
MANO_SCRIPT = PORTABLE_ROOT / "visualize_lerobot_mano.py"
TACTILE_SCRIPT = PORTABLE_ROOT / "visualize_lerobot_tactile.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample episodes from a complete LeRobot dataset and render both tactile and MANO "
            "visualizations for quick spot checks."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python visualization/quick_sample_visualizations.py /path/to/dataset_root\n"
            "  python visualization/quick_sample_visualizations.py /path/to/dataset_root --sample-count 5 --seed 7\n"
            "  python visualization/quick_sample_visualizations.py /path/to/dataset_root --episodes 12 15 18\n"
            "  python visualization/quick_sample_visualizations.py /path/to/dataset_root --max-frames 120 --output_dir /tmp/spot_checks\n"
            "  python visualization/quick_sample_visualizations.py /path/to/dataset_root --episodes 12 --render-combined\n"
            "  python visualization/quick_sample_visualizations.py /path/to/dataset_root "
            "--priority-episodes-file /tmp/risk_scan/priority_episode_indices.txt"
        ),
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Dataset root or dataset data/ directory.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="How many successful random episodes to visualize when --episodes is not provided.",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        metavar="EPISODE",
        help="Optional fixed episode indices. Supports spaces and commas, for example: --episodes 12 15 18 or --episodes 12,15,18",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for episode sampling.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum sampled episodes to try before giving up. Default: all available episodes.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Forwarded to both visualizers. Use 0 to render the full episode.",
    )
    parser.add_argument(
        "--frame-interval",
        "--frame_interval",
        dest="frame_interval",
        type=int,
        default=1,
        help="Forwarded to both visualizers. Render every Nth frame while keeping FPS unchanged, so the saved video plays faster when N > 1.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--output-path",
        "--output_path",
        type=str,
        default="",
        help=(
            "Directory for rendered videos, logs, and the summary JSON. Defaults to "
            "<dataset>/visualizations/spot_checks/<timestamp>, or <dataset>/visualizations/priority_debug/<timestamp> "
            "when --priority-episodes-file is used."
        ),
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to launch the two visualization scripts.",
    )
    parser.add_argument(
        "--mano-hand",
        type=str,
        default="both",
        choices=["left", "right", "both"],
        help="Forwarded to visualize_lerobot_mano.py --hand.",
    )
    parser.add_argument(
        "--mano-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Forwarded to visualize_lerobot_mano.py --device.",
    )
    parser.add_argument(
        "--mano-alpha",
        type=float,
        default=0.35,
        help="Forwarded to visualize_lerobot_mano.py --alpha.",
    )
    parser.add_argument(
        "--tactile-column-prefix",
        type=str,
        default="tactile",
        choices=["tactile", "tactile_pred"],
        help="Forwarded to visualize_lerobot_tactile.py --column_prefix.",
    )
    parser.add_argument(
        "--tactile-panel-width",
        type=int,
        default=640,
        help="Forwarded to visualize_lerobot_tactile.py --panel_width.",
    )
    parser.add_argument(
        "--tactile-panel-height",
        type=int,
        default=320,
        help="Forwarded to visualize_lerobot_tactile.py --panel_height.",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep failed episode directories instead of deleting them when either visualization command fails.",
    )
    parser.add_argument(
        "--render-combined",
        action="store_true",
        help="Also render a side-by-side overview video with MANO on the left and tactile on the right. Automatically enabled in priority debug mode.",
    )
    parser.add_argument(
        "--priority-episodes-file",
        "--priority_episodes_file",
        dest="priority_episodes_file",
        type=str,
        default="",
        help="Ordered episode index list for debug inspection, for example priority_episode_indices.txt from utils/rank_risk_scan_results.py.",
    )
    parser.add_argument(
        "--keep-intermediate-videos",
        action="store_true",
        help="Keep sampled tactile/MANO intermediate videos when using priority debug mode. By default only the combined debug video is kept.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which episodes would be sampled and which commands would run.",
    )
    args = parser.parse_args()
    args.frame_interval_explicit = any(flag in sys.argv[1:] for flag in ("--frame-interval", "--frame_interval"))
    return args


def _normalize_dataset_root(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        raise ValueError(f"Expected a dataset directory, got file: {path}")
    if path.name in {"data", "meta", "videos", "video_backup"} and path.parent.exists():
        path = path.parent
    data_dir = path / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset data/ directory not found under: {path}")
    return path


def _parse_episode_tokens(raw_values: Sequence[str] | None) -> list[int]:
    if not raw_values:
        return []
    episode_indices: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            stripped = token.strip()
            if not stripped:
                continue
            value = int(stripped)
            if value not in seen:
                seen.add(value)
                episode_indices.append(value)
    return episode_indices


def _parse_priority_episode_file(raw_path: str) -> list[int]:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Priority episode file not found: {path}")
    episode_indices: list[int] = []
    seen: set[int] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        tokens: list[str] = []
        for comma_part in stripped.split(","):
            tokens.extend(token for token in comma_part.split() if token)
        for token in tokens:
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid episode token in {path}:{line_number}: {token!r}") from exc
            if value not in seen:
                seen.add(value)
                episode_indices.append(value)
    if not episode_indices:
        raise ValueError(f"No episode indices found in priority file: {path}")
    return episode_indices


def _list_available_episode_indices(dataset_root: Path) -> list[int]:
    episode_indices: list[int] = []
    for parquet_path in sorted((dataset_root / "data").glob("episode_*.parquet")):
        stem = parquet_path.stem
        suffix = stem.removeprefix("episode_")
        if suffix.isdigit():
            episode_indices.append(int(suffix))
    return episode_indices


def _build_output_dir(dataset_root: Path, output_dir_arg: str, *, priority_mode: bool = False) -> Path:
    if output_dir_arg.strip():
        return Path(output_dir_arg).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if priority_mode:
        return (dataset_root / "visualizations" / "priority_debug" / timestamp).resolve()
    return (dataset_root / "visualizations" / "spot_checks" / timestamp).resolve()


def _episode_dir(output_dir: Path, episode_index: int) -> Path:
    return output_dir / f"episode_{int(episode_index):06d}"


def _output_name_suffix(frame_interval: int) -> str:
    interval = max(1, int(frame_interval))
    if interval <= 1:
        return ""
    return f"_interval{interval}"


def _format_missing_episode_message(prefix: str, missing: Sequence[int]) -> str:
    ordered = sorted(int(value) for value in missing)
    preview = ", ".join(str(value) for value in ordered[:20])
    if len(ordered) > 20:
        preview += f", ... (+{len(ordered) - 20} more)"
    return f"{prefix}: {len(ordered)} missing episode(s). Examples: {preview}"


def _build_tactile_command(
    python_exe: str,
    dataset_root: Path,
    episode_index: int,
    output_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(python_exe),
        str(TACTILE_SCRIPT),
        str(dataset_root),
        "--episode",
        str(int(episode_index)),
        "--column_prefix",
        str(args.tactile_column_prefix),
        "--panel_width",
        str(int(args.tactile_panel_width)),
        "--panel_height",
        str(int(args.tactile_panel_height)),
        "-o",
        str(output_path),
    ]
    if int(args.max_frames) >= 0:
        command.extend(["--max_frames", str(int(args.max_frames))])
    command.extend(["--frame_interval", str(max(1, int(args.frame_interval)))])
    return command


def _build_mano_command(
    python_exe: str,
    dataset_root: Path,
    episode_index: int,
    output_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(python_exe),
        str(MANO_SCRIPT),
        str(dataset_root),
        "--episode",
        str(int(episode_index)),
        "--hand",
        str(args.mano_hand),
        "--device",
        str(args.mano_device),
        "--alpha",
        str(float(args.mano_alpha)),
        "-o",
        str(output_path),
    ]
    if int(args.max_frames) >= 0:
        command.extend(["--max_frames", str(int(args.max_frames))])
    command.extend(["--frame_interval", str(max(1, int(args.frame_interval)))])
    return command


def _run_command(command: Sequence[str], log_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(PORTABLE_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    combined_output = ""
    if completed.stdout:
        combined_output += completed.stdout
    if completed.stderr:
        if combined_output:
            combined_output += "\n"
        combined_output += completed.stderr
    log_path.write_text(combined_output, encoding="utf-8")
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "log_path": str(log_path),
    }


def _resize_frame_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise RuntimeError("Encountered an invalid video frame with zero width or height.")
    if source_height == int(target_height):
        return frame
    scale = float(target_height) / float(source_height)
    target_width = max(1, int(round(float(source_width) * scale)))
    interpolation = cv2.INTER_AREA if int(target_height) < source_height else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, int(target_height)), interpolation=interpolation)


def _render_combined_video(
    mano_video_path: Path,
    tactile_video_path: Path,
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    log_lines = [
        f"mano_video={mano_video_path}",
        f"tactile_video={tactile_video_path}",
        f"output_video={output_path}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    mano_capture: cv2.VideoCapture | None = None
    tactile_capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    returncode = 1

    try:
        mano_capture = cv2.VideoCapture(str(mano_video_path))
        if not mano_capture.isOpened():
            raise RuntimeError(f"Failed to open MANO video: {mano_video_path}")
        tactile_capture = cv2.VideoCapture(str(tactile_video_path))
        if not tactile_capture.isOpened():
            raise RuntimeError(f"Failed to open tactile video: {tactile_video_path}")

        mano_width = int(mano_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        mano_height = int(mano_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tactile_width = int(tactile_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        tactile_height = int(tactile_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mano_fps = float(mano_capture.get(cv2.CAP_PROP_FPS))
        tactile_fps = float(tactile_capture.get(cv2.CAP_PROP_FPS))
        mano_frame_count = int(max(0, mano_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        tactile_frame_count = int(max(0, tactile_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        if mano_width <= 0 or mano_height <= 0:
            raise RuntimeError(f"Invalid MANO video shape: {mano_width}x{mano_height}")
        if tactile_width <= 0 or tactile_height <= 0:
            raise RuntimeError(f"Invalid tactile video shape: {tactile_width}x{tactile_height}")

        log_lines.extend(
            [
                f"mano_shape={mano_width}x{mano_height}",
                f"tactile_shape={tactile_width}x{tactile_height}",
                f"mano_fps={mano_fps:.6f}",
                f"tactile_fps={tactile_fps:.6f}",
                f"mano_frame_count={mano_frame_count}",
                f"tactile_frame_count={tactile_frame_count}",
            ]
        )

        target_height = max(mano_height, tactile_height)
        resized_mano_width = max(1, int(round(float(mano_width) * float(target_height) / float(mano_height))))
        resized_tactile_width = max(1, int(round(float(tactile_width) * float(target_height) / float(tactile_height))))
        output_width = resized_mano_width + resized_tactile_width
        output_fps = mano_fps if mano_fps > 0.0 else (tactile_fps if tactile_fps > 0.0 else 30.0)

        if mano_fps > 0.0 and tactile_fps > 0.0 and abs(mano_fps - tactile_fps) > 1e-3:
            log_lines.append(
                "warning=fps_mismatch combining with MANO fps as output "
                f"({mano_fps:.6f} vs {tactile_fps:.6f})"
            )
        if mano_frame_count > 0 and tactile_frame_count > 0 and mano_frame_count != tactile_frame_count:
            log_lines.append(
                "warning=frame_count_mismatch combined output will stop at the shorter source "
                f"({mano_frame_count} vs {tactile_frame_count})"
            )

        log_lines.extend(
            [
                f"target_height={target_height}",
                f"resized_mano_width={resized_mano_width}",
                f"resized_tactile_width={resized_tactile_width}",
                f"output_shape={output_width}x{target_height}",
                f"output_fps={output_fps:.6f}",
            ]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, float(output_fps), (int(output_width), int(target_height)))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open combined output video writer: {output_path}")

        while True:
            mano_ok, mano_frame = mano_capture.read()
            tactile_ok, tactile_frame = tactile_capture.read()
            if not mano_ok or not tactile_ok:
                if mano_ok != tactile_ok:
                    log_lines.append("warning=source_exhausted_early one source ended before the other")
                break
            combined_frame = np.concatenate(
                [
                    _resize_frame_to_height(mano_frame, target_height),
                    _resize_frame_to_height(tactile_frame, target_height),
                ],
                axis=1,
            )
            writer.write(combined_frame)
            frames_written += 1

        if frames_written <= 0:
            raise RuntimeError("Combined video writer produced zero frames.")

        returncode = 0
        log_lines.append(f"frames_written={frames_written}")
        return {
            "returncode": returncode,
            "log_path": str(log_path),
            "output_path": str(output_path),
            "frames_written": int(frames_written),
        }
    except Exception as exc:
        log_lines.append(f"error={type(exc).__name__}: {exc}")
        if output_path.exists():
            output_path.unlink()
        return {
            "returncode": returncode,
            "log_path": str(log_path),
            "output_path": str(output_path),
            "frames_written": int(frames_written),
            "error": str(exc),
        }
    finally:
        if writer is not None:
            writer.release()
        if mano_capture is not None:
            mano_capture.release()
        if tactile_capture is not None:
            tactile_capture.release()
        if returncode == 0 and not any(line.startswith("frames_written=") for line in log_lines):
            log_lines.append(f"frames_written={frames_written}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def _write_skipped_combined_log(log_path: Path, reason: str) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"status=skipped\nreason={reason}\n", encoding="utf-8")
    return {
        "returncode": 1,
        "log_path": str(log_path),
        "frames_written": 0,
        "skipped": True,
        "reason": str(reason),
    }


def _delete_intermediate_videos(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _cleanup_failed_episode_dir(episode_dir: Path) -> None:
    if not episode_dir.exists():
        return
    for child in sorted(episode_dir.rglob("*"), key=lambda path: (len(path.parts), str(path)), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    if episode_dir.exists():
        episode_dir.rmdir()


def main() -> int:
    args = _parse_args()
    if int(args.frame_interval) <= 0:
        raise ValueError("--frame-interval must be positive.")
    dataset_root = _normalize_dataset_root(args.dataset_path)
    available_episode_indices = _list_available_episode_indices(dataset_root)
    if not available_episode_indices:
        raise FileNotFoundError(f"No episode parquet files found under: {dataset_root / 'data'}")

    priority_mode = bool(str(args.priority_episodes_file).strip())
    if priority_mode and not bool(getattr(args, "frame_interval_explicit", False)):
        args.frame_interval = 10
    if priority_mode:
        args.render_combined = True
    if priority_mode and args.episodes:
        raise ValueError("Please use either --episodes or --priority-episodes-file, not both.")

    requested_episodes = _parse_episode_tokens(args.episodes)
    if priority_mode:
        requested_episodes = _parse_priority_episode_file(args.priority_episodes_file)
        missing = sorted(set(requested_episodes) - set(available_episode_indices))
        if missing:
            raise FileNotFoundError(
                _format_missing_episode_message(
                    "Episodes from priority file not found in dataset; the dataset and ranking file may not match",
                    missing,
                )
            )
        candidate_episode_indices = requested_episodes
        target_success_count = len(requested_episodes)
    elif requested_episodes:
        missing = sorted(set(requested_episodes) - set(available_episode_indices))
        if missing:
            raise FileNotFoundError(_format_missing_episode_message("Requested episodes not found in dataset", missing))
        candidate_episode_indices = requested_episodes
        target_success_count = len(requested_episodes)
    else:
        if int(args.sample_count) <= 0:
            raise ValueError("--sample-count must be positive when --episodes is not provided.")
        rng = random.Random(args.seed)
        candidate_episode_indices = list(available_episode_indices)
        rng.shuffle(candidate_episode_indices)
        target_success_count = min(int(args.sample_count), len(candidate_episode_indices))

    max_attempts = int(args.max_attempts) if int(args.max_attempts) > 0 else len(candidate_episode_indices)
    output_dir = _build_output_dir(dataset_root, args.output_dir, priority_mode=priority_mode)
    keep_intermediate_videos = bool(args.keep_intermediate_videos)
    delete_intermediate_after_combine = bool(priority_mode and not keep_intermediate_videos)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "requested_episodes": requested_episodes,
        "available_episode_count": len(available_episode_indices),
        "target_success_count": int(target_success_count),
        "attempted_episodes": [],
        "successful_episodes": [],
        "failed_episodes": [],
        "dry_run": bool(args.dry_run),
        "render_combined": bool(args.render_combined),
        "priority_mode": bool(priority_mode),
        "priority_episodes_file": str(Path(args.priority_episodes_file).expanduser().resolve()) if priority_mode else "",
        "frame_interval": int(args.frame_interval),
        "keep_intermediate_videos": keep_intermediate_videos,
        "delete_intermediate_after_combine": delete_intermediate_after_combine,
        "run_timestamp": datetime.now().isoformat(),
    }

    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Available episodes: {len(available_episode_indices)}")
    print(f"Target successful spot checks: {target_success_count}")
    print(f"Frame interval: {int(args.frame_interval)}")
    if priority_mode:
        print(f"Priority episode file: {Path(args.priority_episodes_file).expanduser().resolve()}")
        print(f"Priority debug mode: combined_only={delete_intermediate_after_combine}")
    elif requested_episodes:
        print(f"Fixed episodes: {requested_episodes}")
    else:
        print(f"Sampling seed: {args.seed}")

    success_count = 0
    attempt_count = 0
    for episode_index in candidate_episode_indices:
        if success_count >= target_success_count or attempt_count >= max_attempts:
            break
        attempt_count += 1
        episode_dir = _episode_dir(output_dir, episode_index)
        output_suffix = _output_name_suffix(int(args.frame_interval))
        tactile_output = episode_dir / f"episode_{int(episode_index):06d}_tactile{output_suffix}.mp4"
        mano_output = episode_dir / f"episode_{int(episode_index):06d}_mano{output_suffix}.mp4"
        combined_output = episode_dir / f"episode_{int(episode_index):06d}_combined{output_suffix}.mp4"
        tactile_log = episode_dir / "tactile.log"
        mano_log = episode_dir / "mano.log"
        combined_log = episode_dir / "combined.log"

        tactile_command = _build_tactile_command(
            python_exe=args.python_exe,
            dataset_root=dataset_root,
            episode_index=episode_index,
            output_path=tactile_output,
            args=args,
        )
        mano_command = _build_mano_command(
            python_exe=args.python_exe,
            dataset_root=dataset_root,
            episode_index=episode_index,
            output_path=mano_output,
            args=args,
        )

        print(f"[Attempt {attempt_count}/{max_attempts}] episode {episode_index}")
        print(f"  tactile -> {tactile_output}")
        print(f"  mano    -> {mano_output}")
        if args.render_combined:
            print(f"  combined-> {combined_output}")

        attempt_record: dict[str, Any] = {
            "episode_index": int(episode_index),
            "episode_dir": str(episode_dir),
            "tactile_output": str(tactile_output),
            "mano_output": str(mano_output),
            "combined_output": str(combined_output) if args.render_combined else "",
            "frame_interval": int(args.frame_interval),
            "priority_mode": bool(priority_mode),
            "intermediate_outputs_deleted": False,
            "tactile_command": tactile_command,
            "mano_command": mano_command,
        }

        if args.dry_run:
            attempt_record["status"] = "dry_run"
            summary["attempted_episodes"].append(attempt_record)
            success_count += 1
            summary["successful_episodes"].append(attempt_record)
            continue

        episode_dir.mkdir(parents=True, exist_ok=True)
        tactile_result = _run_command(tactile_command, tactile_log)
        mano_result = _run_command(mano_command, mano_log)
        attempt_record["tactile_result"] = tactile_result
        attempt_record["mano_result"] = mano_result
        combined_result: dict[str, Any] | None = None
        if args.render_combined:
            tactile_ok = tactile_result["returncode"] == 0
            mano_ok = mano_result["returncode"] == 0
            if tactile_ok and mano_ok:
                combined_result = _render_combined_video(
                    mano_video_path=mano_output,
                    tactile_video_path=tactile_output,
                    output_path=combined_output,
                    log_path=combined_log,
                )
            else:
                combined_result = _write_skipped_combined_log(
                    combined_log,
                    reason="tactile or MANO rendering failed before combined rendering could start",
                )
            attempt_record["combined_result"] = combined_result
            if (
                delete_intermediate_after_combine
                and combined_result is not None
                and combined_result["returncode"] == 0
            ):
                _delete_intermediate_videos([tactile_output, mano_output])
                attempt_record["intermediate_outputs_deleted"] = True

        success = tactile_result["returncode"] == 0 and mano_result["returncode"] == 0
        if args.render_combined:
            success = success and combined_result is not None and combined_result["returncode"] == 0
        attempt_record["status"] = "ok" if success else "failed"
        summary["attempted_episodes"].append(attempt_record)
        if success:
            success_count += 1
            summary["successful_episodes"].append(attempt_record)
            if args.render_combined:
                print(
                    "  success: tactile_log={tactile_log} mano_log={mano_log} combined_log={combined_log}".format(
                        tactile_log=tactile_log,
                        mano_log=mano_log,
                        combined_log=combined_log,
                    )
                )
            else:
                print(
                    "  success: tactile_log={tactile_log} mano_log={mano_log}".format(
                        tactile_log=tactile_log,
                        mano_log=mano_log,
                    )
                )
        else:
            summary["failed_episodes"].append(attempt_record)
            if args.render_combined:
                combined_rc = combined_result["returncode"] if combined_result is not None else "n/a"
                print(
                    "  failed: tactile_rc={tactile_rc} mano_rc={mano_rc} combined_rc={combined_rc}".format(
                        tactile_rc=tactile_result["returncode"],
                        mano_rc=mano_result["returncode"],
                        combined_rc=combined_rc,
                    )
                )
                print(f"  logs: {tactile_log} | {mano_log} | {combined_log}")
            else:
                print(
                    "  failed: tactile_rc={tactile_rc} mano_rc={mano_rc}".format(
                        tactile_rc=tactile_result["returncode"],
                        mano_rc=mano_result["returncode"],
                    )
                )
                print(f"  logs: {tactile_log} | {mano_log}")
            if not args.keep_failed:
                _cleanup_failed_episode_dir(episode_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary["attempt_count"] = int(attempt_count)
    summary["success_count"] = int(success_count)
    summary["requested_success_count"] = int(target_success_count)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Summary written to: {summary_path}")
    if success_count < target_success_count:
        print(
            "Only {success}/{target} successful spot checks were produced.".format(
                success=success_count,
                target=target_success_count,
            )
        )
        return 1

    print(f"Successful spot checks: {success_count}/{target_success_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
