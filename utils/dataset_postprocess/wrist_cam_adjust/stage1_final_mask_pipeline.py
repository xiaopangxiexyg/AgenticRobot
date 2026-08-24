from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deployments.openpi.training.prepared_skill import (
    OpenPIPreparedSkillBinding,
    load_openpi_prepared_skill_binding_manifest,
)
from deployments.openpi.training.prepared_skill_launch import _load_profile

from .depth_mask.core import DEPTH_RANGE_POSTPROCESSED_MASK_NAME
from .depth_mask.postprocess import (
    DepthMaskPostprocessConfig,
    DepthMaskPostprocessResult,
    process_mask_stream,
)
from .depth_mask.visualize import (
    find_session_mask_manifests,
    load_mask_stream,
)
from .final_mask import compose_final_masks
from .geometry_mask.dataset import adjust_dataset

PIPELINE_SCHEMA_V1 = "dexteleop_stage1_final_keep_mask_pipeline_v1"
FINAL_MASK_LAYOUT_V1 = "source_dataset_id_partitioned_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = _read_json(path)
        if current != dict(payload):
            raise FileExistsError(
                f"receipt already exists with different content: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _episode_records(dataset_root: Path) -> dict[int, dict[str, Any]]:
    return {
        int(record["episode_index"]): record
        for record in _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    }


def _selected_by_source(
    binding: OpenPIPreparedSkillBinding,
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for episode in binding.episodes:
        grouped.setdefault(episode.source_dataset_id, []).append(
            int(episode.source_episode_index)
        )
    return {
        source_id: tuple(sorted(indices))
        for source_id, indices in grouped.items()
    }


def _physical_devices_by_episode(
    dataset_root: Path,
    episode_indices: Sequence[int],
) -> dict[int, Path] | None:
    default_path = dataset_root / "meta" / "devices" / "realsense.json"
    if default_path.is_file():
        return None
    audit_path = dataset_root / "meta" / "build_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            "dataset has neither one devices manifest nor a physical build "
            f"audit: {dataset_root}"
        )
    audit = _read_json(audit_path)
    plan = audit.get("plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("episodes"), list):
        raise TypeError(f"physical build audit has no episode plan: {audit_path}")
    selected = {int(value) for value in episode_indices}
    mapping: dict[int, Path] = {}
    for item in plan["episodes"]:
        if not isinstance(item, Mapping):
            continue
        output_index = int(item["output_episode_index"])
        if output_index not in selected:
            continue
        source_root = Path(str(item["source_root"])).expanduser().resolve()
        devices_path = (
            dataset_root
            / "meta"
            / "source_devices"
            / source_root.name
            / "realsense.json"
        ).resolve()
        original_path = source_root / "meta" / "devices" / "realsense.json"
        if not devices_path.is_file() or not original_path.is_file():
            raise FileNotFoundError(
                "physical source devices manifest is missing for output "
                f"episode {output_index}: {devices_path}"
            )
        if _sha256(devices_path) != _sha256(original_path):
            raise ValueError(
                "physical source devices copy differs from the original for "
                f"output episode {output_index}."
            )
        mapping[output_index] = devices_path
    missing = sorted(selected - set(mapping))
    if missing:
        raise ValueError(
            "physical build audit does not bind every selected episode to "
            f"camera intrinsics: {missing[:8]}"
        )
    return mapping


def _camera_records(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("cameras")
    return nested if isinstance(nested, Mapping) else payload


def _camera_serial(record: Mapping[str, Any]) -> str:
    return str(record.get("serial_number") or record.get("serial") or "").strip()


def _intrinsics_overlay_for_source(
    *,
    profile: Mapping[str, Any],
    profile_sources: Mapping[str, Path],
    source_id: str,
    dataset_root: Path,
    overlay_base: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    overrides = profile.get("right_wrist_intrinsics_overrides", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("right_wrist_intrinsics_overrides must be an object")
    raw_override = overrides.get(source_id)
    if raw_override is None:
        return None, None
    if not isinstance(raw_override, Mapping):
        raise TypeError(f"intrinsics override must be an object: {source_id}")
    target_camera_key = str(raw_override.get("target_camera_key", "")).strip()
    donor_source_id = str(
        raw_override.get("donor_source_dataset_id", "")
    ).strip()
    donor_camera_key = str(raw_override.get("donor_camera_key", "")).strip()
    expected_serial = str(raw_override.get("expected_serial", "")).strip()
    if (
        target_camera_key != "right_wrist"
        or donor_source_id not in profile_sources
        or not donor_camera_key
        or not expected_serial
    ):
        raise ValueError(f"invalid right-wrist intrinsics override: {source_id}")

    target_path = dataset_root / "meta" / "devices" / "realsense.json"
    donor_path = (
        profile_sources[donor_source_id]
        / "meta"
        / "devices"
        / "realsense.json"
    )
    target_payload = _read_json(target_path)
    donor_payload = _read_json(donor_path)
    target_record = _camera_records(target_payload).get(target_camera_key)
    donor_record = _camera_records(donor_payload).get(donor_camera_key)
    if not isinstance(target_record, Mapping) or not isinstance(
        donor_record,
        Mapping,
    ):
        raise TypeError(
            f"intrinsics override camera binding is absent: {source_id}"
        )
    target_serial = _camera_serial(target_record)
    donor_serial = _camera_serial(donor_record)
    intrinsics = donor_record.get("intrinsics")
    if (
        target_serial != expected_serial
        or donor_serial != expected_serial
        or not isinstance(intrinsics, Mapping)
    ):
        raise ValueError(
            "intrinsics donor does not match the target camera serial: "
            f"source={source_id} target={target_serial!r} "
            f"donor={donor_serial!r} expected={expected_serial!r}"
        )
    overlay_path = overlay_base / source_id / "realsense.json"
    contract = {
        "schema": "dexteleop.serial_bound_intrinsics_overlay_v1",
        "source_dataset_id": source_id,
        "target_devices_path": str(target_path),
        "target_devices_sha256": _sha256(target_path),
        "target_camera_key": target_camera_key,
        "donor_source_dataset_id": donor_source_id,
        "donor_devices_path": str(donor_path),
        "donor_devices_sha256": _sha256(donor_path),
        "donor_camera_key": donor_camera_key,
        "bound_serial": expected_serial,
        "source_modified": False,
    }
    overlay_payload = {
        "schema": "dexteleop.serial_bound_intrinsics_overlay.v1",
        target_camera_key: {
            "camera_key": target_camera_key,
            "semantic_role": target_camera_key,
            "serial": expected_serial,
            "serial_number": expected_serial,
            "intrinsics": dict(intrinsics),
        },
        "calibration_provenance": contract,
    }
    _write_receipt(overlay_path, overlay_payload)
    return overlay_path, contract


def _expected_frames(
    records: Mapping[int, Mapping[str, Any]],
    episodes: Sequence[int],
    *,
    max_frames: int,
) -> dict[int, int]:
    expected: dict[int, int] = {}
    for episode_index in episodes:
        record = records.get(int(episode_index))
        if record is None:
            raise ValueError(f"selected episode is absent: {episode_index}")
        frame_count = int(record.get("frame_count", 0))
        if frame_count <= 0:
            raise ValueError(
                f"selected episode has no positive frame_count: {episode_index}"
            )
        expected[int(episode_index)] = (
            min(frame_count, int(max_frames))
            if int(max_frames) > 0
            else frame_count
        )
    return expected


def _verify_mask_tree(
    root: Path,
    dataset_name: str,
    expected_frames: Mapping[int, int],
    *,
    stream_name: str,
    expected_mask_name: str,
) -> dict[str, Any]:
    total_frames = 0
    total_bytes = 0
    for episode_index, frame_count in sorted(expected_frames.items()):
        episode_dir = (
            root
            / dataset_name
            / f"episode_{episode_index:06d}"
            / "right_wrist"
        )
        manifest_path = episode_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("mask_name") != expected_mask_name:
            raise ValueError(f"unexpected mask semantics: {manifest_path}")
        stream = manifest.get("stream")
        if not isinstance(stream, Mapping):
            raise TypeError(f"mask manifest has no stream: {manifest_path}")
        actual_count = int(stream.get("frame_count", 0))
        if actual_count != frame_count:
            raise ValueError(
                "mask frame count differs: "
                f"episode={episode_index} actual={actual_count} "
                f"expected={frame_count}"
            )
        packed = int(stream.get("packed_bytes_per_frame", 0))
        stream_path = episode_dir / stream_name
        actual_bytes = int(stream_path.stat().st_size)
        expected_bytes = actual_count * packed
        if actual_bytes != expected_bytes:
            raise ValueError(
                "mask stream size differs: "
                f"episode={episode_index} actual={actual_bytes} "
                f"expected={expected_bytes}"
            )
        total_frames += actual_count
        total_bytes += actual_bytes
    return {
        "episode_count": len(expected_frames),
        "frame_count": total_frames,
        "stream_bytes": total_bytes,
    }


def _postprocess_source(
    dataset_root: Path,
    output_root: Path,
    episodes: Sequence[int],
    *,
    jobs: int,
    max_frames: int,
) -> tuple[list[DepthMaskPostprocessResult], list[dict[str, Any]]]:
    config = DepthMaskPostprocessConfig()

    def run(episode_index: int) -> DepthMaskPostprocessResult:
        manifests = find_session_mask_manifests(
            dataset_root,
            episode=episode_index,
            camera_keys=("right_wrist",),
        )
        if len(manifests) != 1:
            raise ValueError(
                f"episode {episode_index} did not resolve one right wrist mask"
            )
        stream = load_mask_stream(manifests[0], root_hint=dataset_root)
        output_dir = (
            output_root
            / dataset_root.name
            / f"episode_{episode_index:06d}"
            / "right_wrist"
        )
        return process_mask_stream(
            stream,
            output_dir,
            config=config,
            max_frames=max_frames,
            overwrite=output_dir.exists(),
        )

    results: list[DepthMaskPostprocessResult] = []
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as executor:
        futures = {executor.submit(run, episode): episode for episode in episodes}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: str(item.manifest_path))
    trailing = [
        {
            "manifest_path": str(result.manifest_path),
            "source_declared_bytes": result.source_declared_bytes,
            "source_actual_bytes": result.source_actual_bytes,
            "source_trailing_bytes": result.source_trailing_bytes,
        }
        for result in results
        if result.source_trailing_bytes
    ]
    return results, trailing


def _source_receipt(
    *,
    profile_path: Path,
    binding: OpenPIPreparedSkillBinding,
    source_id: str,
    dataset_root: Path,
    episodes: Sequence[int],
    expected_frames: Mapping[int, int],
    table_root: Path,
    post_root: Path,
    final_root: Path,
    trailing_sources: Sequence[Mapping[str, Any]],
    calibration_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": PIPELINE_SCHEMA_V1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_path": str(profile_path),
        "profile_sha256": _sha256(profile_path),
        "binding_hash": binding.binding_hash,
        "source_dataset_id": source_id,
        "source_dataset_root": str(dataset_root),
        "eligible_episode_count": len(episodes),
        "eligible_frame_count": sum(expected_frames.values()),
        "episode_indices": list(episodes),
        "camera": "right_wrist",
        "mask_policy": (
            "raw depth keep -> default spatial/temporal postprocess -> "
            "OR table geometry keep -> final keep mask"
        ),
        "final_mask_root_layout": FINAL_MASK_LAYOUT_V1,
        "raw_source_modified": False,
        "raw_trailing_streams": [dict(item) for item in trailing_sources],
        "calibration_contract": (
            None
            if calibration_contract is None
            else dict(calibration_contract)
        ),
        "derived_roots": {
            "table_geometry_keep": str(table_root),
            "postprocessed_depth_keep": str(post_root),
            "final_keep": str(final_root),
        },
        "rollback": {
            "policy": "remove only the three source-partitioned derived roots",
            "raw_source_restoration_required": False,
        },
    }


def _resolve_pipeline_derived_roots(
    *,
    configured_final_root: Path,
    workspace_root: Path | None,
    intermediate_workspace_root: Path | None,
) -> tuple[Path, Path]:
    final_derived_root = (
        configured_final_root.parent
        if workspace_root is None
        else workspace_root.expanduser().resolve() / "derived"
    )
    final_base = final_derived_root / "final_keep_masks"
    if workspace_root is None and final_base != configured_final_root:
        raise ValueError("profile final-mask path is not under one derived root")
    intermediate_derived_root = (
        final_derived_root
        if intermediate_workspace_root is None
        else intermediate_workspace_root.expanduser().resolve() / "derived"
    )
    return final_derived_root, intermediate_derived_root


def run_pipeline(
    *,
    profile_path: Path,
    binding_root: Path,
    urdf_path: Path,
    workspace_root: Path | None,
    intermediate_workspace_root: Path | None = None,
    source_ids: Sequence[str],
    jobs: int,
    max_frames: int,
    verify_only: bool,
) -> dict[str, Any]:
    profile = _load_profile(profile_path)
    if profile.get("openpi_final_mask_root_layout") != FINAL_MASK_LAYOUT_V1:
        raise ValueError("profile does not declare source-partitioned final masks")
    binding = load_openpi_prepared_skill_binding_manifest(binding_root)
    selected = _selected_by_source(binding)
    profile_sources = {
        str(item["source_dataset_id"]): Path(str(item["dataset_root"])).resolve()
        for item in profile["sources"]
    }
    if set(selected) != set(profile_sources):
        raise ValueError("binding sources differ from the Stage1 profile")
    requested = set(source_ids)
    if requested:
        unknown = sorted(requested - set(profile_sources))
        if unknown:
            raise ValueError(f"unknown source IDs: {unknown}")
    run_sources = [
        source_id
        for source_id in profile_sources
        if not requested or source_id in requested
    ]

    configured_final_root = Path(str(profile["openpi_final_mask_root"])).resolve()
    derived_root, intermediate_derived_root = _resolve_pipeline_derived_roots(
        configured_final_root=configured_final_root,
        workspace_root=workspace_root,
        intermediate_workspace_root=intermediate_workspace_root,
    )
    final_base = derived_root / "final_keep_masks"
    table_base = intermediate_derived_root / "table_geometry_keep_masks"
    post_base = intermediate_derived_root / "postprocessed_depth_keep_masks"
    receipt_base = derived_root.parent / "receipts" / "stage1_final_masks"
    overlay_base = intermediate_derived_root / "calibration_overlays"

    summaries: list[dict[str, Any]] = []
    for source_id in run_sources:
        dataset_root = profile_sources[source_id]
        episodes = selected[source_id]
        records = _episode_records(dataset_root)
        expected = _expected_frames(records, episodes, max_frames=max_frames)
        table_root = table_base / source_id
        post_root = post_base / source_id
        final_root = final_base / source_id
        receipt_path = receipt_base / f"{source_id}.json"
        existing_receipt = (
            _read_json(receipt_path) if receipt_path.is_file() else None
        )
        if existing_receipt is not None:
            expected_receipt_identity = {
                "schema": PIPELINE_SCHEMA_V1,
                "binding_hash": binding.binding_hash,
                "source_dataset_id": source_id,
                "source_dataset_root": str(dataset_root),
                "eligible_episode_count": len(episodes),
                "eligible_frame_count": sum(expected.values()),
            }
            mismatched = {
                key: (existing_receipt.get(key), expected_value)
                for key, expected_value in expected_receipt_identity.items()
                if existing_receipt.get(key) != expected_value
            }
            if mismatched:
                raise ValueError(
                    "existing source receipt differs from the requested fixed "
                    f"snapshot: {receipt_path} {mismatched}"
                )

        calibration_contract = (
            None
            if existing_receipt is None
            else existing_receipt.get("calibration_contract")
        )
        if not verify_only and existing_receipt is None:
            override_devices, calibration_contract = (
                _intrinsics_overlay_for_source(
                    profile=profile,
                    profile_sources=profile_sources,
                    source_id=source_id,
                    dataset_root=dataset_root,
                    overlay_base=overlay_base,
                )
            )
            devices_by_episode = _physical_devices_by_episode(
                dataset_root,
                episodes,
            )
            adjust_dataset(
                dataset_root,
                table_root,
                urdf_path=urdf_path,
                cameras=("right_wrist",),
                episodes=episodes,
                devices_json=override_devices,
                devices_json_by_episode=devices_by_episode,
                output_kind="keep-mask",
                table_mask_mode="sampled-raycast",
                table_raycast_stride=2,
                copy_mode="skip",
                jobs=jobs,
                overwrite=table_root.exists(),
                max_frames=max_frames,
                show_progress=True,
            )
            _post_results, trailing = _postprocess_source(
                dataset_root,
                post_root,
                episodes,
                jobs=jobs,
                max_frames=max_frames,
            )
            compose_final_masks(
                session_root=dataset_root,
                table_mask_root=table_root,
                depth_mask_root=post_root,
                output_root=final_root,
                episodes=episodes,
                cameras=("right_wrist",),
                max_frames=max_frames,
                jobs=jobs,
                overwrite=(final_root / dataset_root.name).exists(),
            )
        else:
            trailing = list(
                (existing_receipt or {}).get("raw_trailing_streams", [])
            )

        post_summary = _verify_mask_tree(
            post_root,
            dataset_root.name,
            expected,
            stream_name="mask_postprocessed.bitpack.bin",
            expected_mask_name=DEPTH_RANGE_POSTPROCESSED_MASK_NAME,
        )
        final_summary = _verify_mask_tree(
            final_root,
            dataset_root.name,
            expected,
            stream_name="final_keep.bitpack.bin",
            expected_mask_name="postprocessed_depth_or_table_keep_mask",
        )
        receipt = _source_receipt(
            profile_path=profile_path,
            binding=binding,
            source_id=source_id,
            dataset_root=dataset_root,
            episodes=episodes,
            expected_frames=expected,
            table_root=table_root,
            post_root=post_root,
            final_root=final_root,
            trailing_sources=trailing,
            calibration_contract=(
                calibration_contract
                if isinstance(calibration_contract, Mapping)
                else None
            ),
        )
        if not verify_only and existing_receipt is None:
            _write_receipt(receipt_path, receipt)
        summaries.append(
            {
                "source_dataset_id": source_id,
                "dataset_root": str(dataset_root),
                "postprocessed": post_summary,
                "final": final_summary,
                "raw_trailing_stream_count": len(trailing),
                "receipt_path": str(receipt_path),
            }
        )

    summary = {
        "schema": PIPELINE_SCHEMA_V1,
        "profile_path": str(profile_path),
        "binding_root": str(binding_root),
        "binding_hash": binding.binding_hash,
        "workspace_derived_root": str(derived_root),
        "intermediate_workspace_derived_root": str(intermediate_derived_root),
        "source_count": len(summaries),
        "eligible_episode_count": sum(
            int(item["final"]["episode_count"]) for item in summaries
        ),
        "eligible_frame_count": sum(
            int(item["final"]["frame_count"]) for item in summaries
        ),
        "sources": summaries,
    }
    if not verify_only:
        _write_receipt(receipt_base / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and verify source-partitioned Stage1 right-wrist final "
            "keep masks without modifying source datasets."
        )
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--binding-root", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--intermediate-workspace-root", type=Path, default=None)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    summary = run_pipeline(
        profile_path=args.profile.expanduser().resolve(),
        binding_root=args.binding_root.expanduser().resolve(),
        urdf_path=args.urdf.expanduser().resolve(),
        workspace_root=args.workspace_root,
        intermediate_workspace_root=args.intermediate_workspace_root,
        source_ids=tuple(str(value) for value in args.source_id),
        jobs=args.jobs,
        max_frames=args.max_frames,
        verify_only=args.verify_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
