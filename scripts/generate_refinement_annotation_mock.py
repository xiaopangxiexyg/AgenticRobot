"""Generate a sealed, stage-profiled mock dataset for annotation UI testing.

This is an offline probe, not a production collection entrypoint.  It uses the
same profile catalog, RealTeleop progress adapter, annotation application, and
dataset-interface discovery path as the live workflow while producing no
hardware or control side effects.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import uuid

from collection_runtime.progress.profiles import (
    bind_dataset_progress_profile,
    bind_dataset_task_stage_profile,
    load_managed_task_profile,
)
from collection_runtime.sample_acceptance import AcceptedSampleReceipt
from control_station.refinement_annotation.application import (
    RefinementAnnotationApplication,
)
from dataset_contracts.causal_snapshot import LOCAL_WALL_CLOCK
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_interface.progress_annotation_jobs import (
    list_progress_annotation_work_items,
)
from teleop_runtime.real_teleop.progress_adapter import (
    RealTeleopProgressEpisode,
)


DEFAULT_PROFILE_ID = (
    "test_dmtac_rm65_bimanual_position_v1_three_stage_progress"
)
CAMERA_KEYS = ("front", "left_wrist", "right_wrist")
FPS = 15
MOCK_BASE_TIMESTAMP = 1785456000.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an isolated mock dataset for the fine-stage annotation GUI."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh output dataset directory; an existing path is rejected.",
    )
    parser.add_argument(
        "--profile-id",
        default=DEFAULT_PROFILE_ID,
        help="Managed LINEAR progress profile with an ordered-stage profile.",
    )
    parser.add_argument(
        "--background-root",
        type=Path,
        default=Path("utils/random_background"),
        help="Directory containing PNG/JPEG mock video sources.",
    )
    parser.add_argument("--seed", type=int, default=2502)
    return parser


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result


def _backgrounds(root: Path) -> tuple[Path, ...]:
    resolved = root.expanduser().resolve()
    values = tuple(
        sorted(
            path
            for path in resolved.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    )
    if len(values) < 3:
        raise ValueError(
            f"At least three background images are required under {resolved}."
        )
    return values


def _video_frame_count(path: Path) -> int:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe did not return a frame count for {path}."
        ) from exc


def _validate_video(path: Path, *, expected_frames: int) -> None:
    declared = _video_frame_count(path)
    if declared != expected_frames:
        raise RuntimeError(
            f"Video frame mismatch for {path}: "
            f"decoded={declared}, expected={expected_frames}."
        )
    full_decode = _run(
        ["ffmpeg", "-v", "warning", "-i", str(path), "-f", "null", "-"]
    )
    if full_decode.stderr.strip():
        raise RuntimeError(
            f"Decoder warning while validating {path}:\n"
            f"{full_decode.stderr.strip()}"
        )
    tail_decode = _run(
        [
            "ffmpeg",
            "-v",
            "warning",
            "-sseof",
            "-0.5",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
    )
    if tail_decode.stderr.strip():
        raise RuntimeError(
            f"Tail decoder warning while validating {path}:\n"
            f"{tail_decode.stderr.strip()}"
        )


def _write_video(
    path: Path,
    *,
    frame_count: int,
    backgrounds: tuple[Path, ...],
    rng: random.Random,
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    source = rng.choice(backgrounds)
    temporary = path.with_name(f".{path.stem}.building.mp4")
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-i",
            str(source),
            "-frames:v",
            str(frame_count),
            "-vf",
            (
                "scale=640:360:force_original_aspect_ratio=increase,"
                "crop=640:360,format=yuv420p"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    _validate_video(temporary, expected_frames=frame_count)
    os.replace(temporary, path)


def _write_parquet(path: Path, *, timestamps: list[float]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building")
    pq.write_table(
        pa.table(
            {
                "timestamp": timestamps,
                "frame_index": list(range(len(timestamps))),
            }
        ),
        temporary,
        compression="zstd",
    )
    table = pq.read_table(temporary)
    if table.num_rows != len(timestamps):
        raise RuntimeError(f"Parquet row-count validation failed for {path}.")
    os.replace(temporary, path)


def _identity(index: int, *, prepared_at: float) -> EpisodeCaptureIdentity:
    suffix = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"dexteleop-refinement-mock:{index}",
    ).hex[:12]
    return EpisodeCaptureIdentity(
        episode_uuid=f"mock-progress-episode-{index:02d}-{suffix}",
        legacy_episode_index=index,
        lifecycle_owner_id="offline_refinement_annotation_mock",
        prepared_at=prepared_at,
        prepared_clock_domain_id=LOCAL_WALL_CLOCK,
    )


def _failure_then_retry(
    episode: RealTeleopProgressEpisode,
    *,
    episode_index: int,
    base: float,
) -> None:
    episode.advance(
        intent_id="prepare-complete",
        effective_at=base + 5.0,
    )
    effect = episode.report_failure(
        intent_id="manipulate-failure",
        effective_at=base + 8.0,
        available_at=base + 8.02,
    )
    episode.record_applied_hold_pause(
        effect_request=effect,
        requested_at=base + 8.0,
        application_hold_effective_at=base + 8.03,
        formal_capture_pause_effective_at=base + 8.04,
        available_at=base + 8.05,
    )
    episode.record_applied_resume(
        resume_operation_id="manipulate-retry",
        requested_at=base + 27.8,
        application_resume_effective_at=base + 28.0,
        formal_capture_resume_effective_at=base + 28.0,
        available_at=base + 28.02,
    )
    episode.retry_on_accepted_sample(
        AcceptedSampleReceipt(
            episode_index=episode_index,
            sample_index=121,
            episode_frame_index=121,
            sample_timestamp=base + 28.0666666667,
        ),
        available_at=base + 28.08,
    )
    episode.advance(
        intent_id="manipulate-retry-complete",
        effective_at=base + 33.0,
    )
    episode.terminate(
        reason="manual_key_stop",
        effective_at=base + 37.0,
    )


def _build_progress_episode(
    root: Path,
    *,
    index: int,
    profile,
    stage_profile,
    base: float,
) -> tuple[RealTeleopProgressEpisode, float]:
    identity = _identity(index, prepared_at=base - 0.1)
    if index == 3:
        episode = RealTeleopProgressEpisode(
            dataset_root=root,
            identity=identity,
            profile=profile,
            stage_profile=stage_profile,
            entry_step_id="test_manipulate",
            entry_stage_id="test_manipulate.execute",
            entry_condition="RECOVERY_FROM_FAILURE",
        )
        duration = 14.0
        episode.bind(started_at=base)
        episode.advance(
            intent_id="recovery-manipulate-complete",
            effective_at=base + 7.0,
        )
        episode.terminate(
            reason="manual_key_stop",
            effective_at=base + duration,
        )
    elif index == 2:
        duration = 13.0
        episode = RealTeleopProgressEpisode(
            dataset_root=root,
            identity=identity,
            profile=profile,
            stage_profile=stage_profile,
        )
        episode.bind(started_at=base)
        episode.advance(
            intent_id="prepare-complete",
            effective_at=base + 5.0,
        )
        episode.terminate(
            reason="quit_requested",
            effective_at=base + duration,
        )
    elif index == 1:
        duration = 37.0
        episode = RealTeleopProgressEpisode(
            dataset_root=root,
            identity=identity,
            profile=profile,
            stage_profile=stage_profile,
        )
        episode.bind(started_at=base)
        _failure_then_retry(
            episode,
            episode_index=index,
            base=base,
        )
    else:
        duration = 17.0
        episode = RealTeleopProgressEpisode(
            dataset_root=root,
            identity=identity,
            profile=profile,
            stage_profile=stage_profile,
        )
        episode.bind(started_at=base)
        episode.advance(
            intent_id="prepare-complete",
            effective_at=base + 5.0,
        )
        episode.advance(
            intent_id="manipulate-complete",
            effective_at=base + 11.0,
        )
        episode.terminate(
            reason="manual_key_stop",
            effective_at=base + duration,
        )
    episode.seal(sealed_at=base + duration + 0.1)
    return episode, duration


def _randomize_draft(
    detail: dict[str, object],
    *,
    rng: random.Random,
) -> dict[str, object]:
    draft = dict(detail["draft"])
    occurrences = [
        dict(value) for value in draft.get("occurrences", [])
    ]
    attempts = {
        str(value["attempt_id"]): value
        for value in detail.get("attempts", [])
    }
    by_attempt: dict[str, list[dict[str, object]]] = defaultdict(list)
    for occurrence in occurrences:
        by_attempt[str(occurrence["coarse_attempt_id"])].append(occurrence)
    for attempt_id, values in by_attempt.items():
        values.sort(key=lambda value: int(value["occurrence_index_in_attempt"]))
        attempt = attempts[attempt_id]
        start_at = float(attempt["start_at"])
        end_at = float(attempt["end_at"])
        weights = [rng.uniform(0.65, 1.35) for _ in values]
        total = sum(weights)
        cursor = start_at
        for index, (occurrence, weight) in enumerate(zip(values, weights)):
            next_cursor = (
                end_at
                if index == len(values) - 1
                else cursor + ((end_at - start_at) * weight / total)
            )
            occurrence["start_at"] = cursor
            occurrence["end_at"] = next_cursor
            occurrence["confidence"] = round(rng.uniform(0.86, 0.99), 3)
            cursor = next_cursor
    draft["occurrences"] = occurrences
    draft["draft_revision"] = int(draft.get("draft_revision", 0)) + 1
    return draft


def _write_readme(root: Path, *, profile_id: str) -> None:
    text = f"""# Offline refinement-annotation mock dataset

This directory is synthetic and must not be used for robot-policy training.

- Profile: `{profile_id}`
- Episode 0: full success; one randomized stage annotation set is sealed.
- Episode 1: failure in the manipulate step, Hold/Pause, then a successful
  retry; its 20-second wall-time Hold has no formal samples, and one randomized
  stage annotation set is sealed.
- Episode 2: partial trajectory ending during the manipulate step.
- Episode 3: `RECOVERY_FROM_FAILURE` entry at `test_manipulate.execute`.
- All Episodes retain randomized editable stage-annotation drafts.
- Videos are generated from `utils/random_background` and contain no robot data.
"""
    (root / "MOCK_DATASET_README.md").write_text(text, encoding="utf-8")


def generate(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output}"
        )
    background_paths = _backgrounds(args.background_root)
    bundle = load_managed_task_profile(args.profile_id)
    if bundle.topology_kind != "LINEAR" or bundle.stage_profile is None:
        raise ValueError(
            "The mock generator requires a managed LINEAR profile with stages."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []
    try:
        bind_dataset_progress_profile(staging, bundle.progress_profile)
        bind_dataset_task_stage_profile(
            staging,
            progress_profile=bundle.progress_profile,
            stage_profile=bundle.stage_profile,
        )
        for index in range(4):
            base = MOCK_BASE_TIMESTAMP + (index * 100.0)
            episode, duration = _build_progress_episode(
                staging,
                index=index,
                profile=bundle.progress_profile,
                stage_profile=bundle.stage_profile,
                base=base,
            )
            if index == 1:
                timestamps = [
                    base + (frame_index / FPS)
                    for frame_index in range((8 * FPS) + 1)
                ]
                retry_start = base + 28.0666666667
                timestamps.extend(
                    retry_start + (frame_index / FPS)
                    for frame_index in range((int((37.0 - 28.0) * FPS)))
                )
            else:
                timestamps = [
                    base + (frame_index / FPS)
                    for frame_index in range(int(duration * FPS) + 1)
                ]
            frame_count = len(timestamps)
            relative_parquet = Path(
                f"data/chunk-000/episode_{index:06d}.parquet"
            )
            _write_parquet(
                staging / relative_parquet,
                timestamps=timestamps,
            )
            videos: dict[str, str] = {}
            for camera in CAMERA_KEYS:
                relative_video = Path(
                    f"videos/observation.images.{camera}/chunk-000/"
                    f"episode_{index:06d}.mp4"
                )
                _write_video(
                    staging / relative_video,
                    frame_count=frame_count,
                    backgrounds=background_paths,
                    rng=rng,
                )
                videos[camera] = relative_video.as_posix()
            records.append(
                {
                    "episode_index": index,
                    "episode_id": f"episode_{index:06d}",
                    "frame_count": frame_count,
                    "start_timestamp": timestamps[0],
                    "end_timestamp": timestamps[-1],
                    "duration_sec": timestamps[-1] - timestamps[0],
                    "parquet_path": relative_parquet.as_posix(),
                    "video_paths": videos,
                    "metadata": {
                        "source_entrypoint": (
                            "scripts.generate_refinement_annotation_mock"
                        ),
                        "mock_dataset": True,
                        "progress_episode": episode.metadata_reference,
                    },
                }
            )
        catalog = staging / "meta" / "episodes.jsonl"
        catalog.parent.mkdir(parents=True, exist_ok=True)
        with catalog.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        items = list_progress_annotation_work_items(staging)
        if len(items) != len(records):
            raise RuntimeError(
                f"Annotation discovery found {len(items)} of {len(records)} "
                "mock Episodes."
            )
        application = RefinementAnnotationApplication(
            staging,
            annotator_id="mock_generator",
        )
        try:
            for index, item in enumerate(items):
                detail = application.episode_detail(item.episode_uuid)
                draft = _randomize_draft(detail, rng=rng)
                application.save_draft(item.episode_uuid, draft)
                if index in {0, 1}:
                    application.commit(item.episode_uuid, draft)
        finally:
            application.close()
        _write_readme(staging, profile_id=bundle.profile_id)
        final_items = list_progress_annotation_work_items(staging)
        if len(final_items) != 4:
            raise RuntimeError("Final annotation discovery validation failed.")
        os.replace(staging, output)
        return {
            "dataset_root": str(output),
            "profile_id": bundle.profile_id,
            "episode_count": len(final_items),
            "sealed_annotation_sets": sum(
                item.annotation_set_count for item in final_items
            ),
            "draft_count": sum(item.has_draft for item in final_items),
            "frame_counts": [item.frame_count for item in final_items],
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = _parser().parse_args()
    started = time.monotonic()
    result = generate(args)
    result["elapsed_sec"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
