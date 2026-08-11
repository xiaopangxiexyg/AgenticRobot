"""Strict read and optional source verification for prepared views."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import (
    canonical_content_hash,
    canonical_loads,
)
from dataset_contracts.progress_skill_projection import (
    ProgressSkillProjectionV1,
    SkillProjectionReceiptV1,
    SkillTrainingContractV1,
)

from ..episode_curation import source_dataset_base_fingerprint
from .io import read_cjsonl, sha256_file
from .models import (
    ANCHORS_FILE,
    DERIVED_FILES,
    EPISODES_FILE,
    INTERVALS_FILE,
    MANIFEST_FILE,
    PREPARED_SELECTED_EPISODE_DOMAIN_V1,
    PREPARED_SKILL_PROJECTION_RECEIPTS_DOMAIN_V1,
    PREPARED_SOURCE_DATASETS_DOMAIN_V1,
    PREPARED_SOURCE_INVENTORY_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_HASH_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_MANIFEST_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_SCHEMA_V1,
    SELECTED_EPISODES_FILE,
    SOURCE_DATASETS_FILE,
    SOURCE_INVENTORY_FILE,
    SKILL_INTERVALS_FILE,
    SKILL_PROJECTION_RECEIPTS_FILE,
    LoadedPreparedTrainingView,
    PreparedTrainingViewError,
    arrow_schemas,
    skill_interval_arrow_schema,
)


def _load_manifest(root: Path) -> Mapping[str, Any]:
    try:
        manifest = canonical_loads(
            (root / MANIFEST_FILE).read_bytes(),
            expected_domain=PREPARED_TRAINING_VIEW_MANIFEST_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PreparedTrainingViewError(
            f"Cannot read prepared-view manifest: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise PreparedTrainingViewError("Prepared-view manifest is not an object.")
    if manifest.get("schema") != PREPARED_TRAINING_VIEW_SCHEMA_V1:
        raise PreparedTrainingViewError("Unsupported prepared-view schema.")
    declared_hash = str(manifest.get("training_view_hash", ""))
    hash_payload = dict(manifest)
    hash_payload.pop("training_view_hash", None)
    actual_hash = canonical_content_hash(
        hash_payload,
        domain=PREPARED_TRAINING_VIEW_HASH_DOMAIN_V1,
    )
    if declared_hash != actual_hash or root.name != actual_hash:
        raise PreparedTrainingViewError("Prepared-view identity hash mismatch.")
    return manifest


def _verify_derived_files(
    root: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    derived = manifest.get("derived_files")
    expected = list(DERIVED_FILES)
    if manifest.get("skill_projection") is not None:
        expected.extend(
            (SKILL_INTERVALS_FILE, SKILL_PROJECTION_RECEIPTS_FILE)
        )
    if not isinstance(derived, Mapping) or set(derived) != set(expected):
        raise PreparedTrainingViewError("Prepared-view derived file set is invalid.")
    result: dict[str, Mapping[str, Any]] = {}
    for filename in expected:
        metadata = derived[filename]
        path = root / filename
        if not isinstance(metadata, Mapping) or not path.is_file():
            raise PreparedTrainingViewError(
                f"Prepared-view file is absent or unbound: {filename}."
            )
        if (
            int(metadata.get("size_bytes", -1)) != path.stat().st_size
            or str(metadata.get("sha256", "")) != sha256_file(path)
        ):
            raise PreparedTrainingViewError(
                f"Prepared-view file hash/size mismatch: {filename}."
            )
        result[filename] = metadata
    return result


def _load_sources(
    *,
    root: Path,
    organized_root: Path,
    manifest: Mapping[str, Any],
    require_self_contained: bool,
) -> tuple[dict[str, Path], dict[str, str]]:
    source_payload = canonical_loads(
        (root / SOURCE_DATASETS_FILE).read_bytes(),
        expected_domain=PREPARED_SOURCE_DATASETS_DOMAIN_V1,
    )
    if not isinstance(source_payload, Mapping):
        raise PreparedTrainingViewError("source_datasets.cjson is invalid.")
    source_values = source_payload.get("sources")
    if not isinstance(source_values, (tuple, list)):
        raise PreparedTrainingViewError("source_datasets.cjson has no source list.")
    source_paths: dict[str, Path] = {}
    source_fingerprints: dict[str, str] = {}
    for value in source_values:
        if not isinstance(value, Mapping):
            raise PreparedTrainingViewError("Source dataset record is invalid.")
        source_id = str(value.get("source_dataset_id", "")).strip()
        locator = str(value.get("source_root_locator", "")).strip()
        pure = PurePosixPath(locator)
        if not source_id or not locator or pure.is_absolute():
            raise PreparedTrainingViewError("Source locator must be relative.")
        resolved = (organized_root / Path(*pure.parts)).resolve()
        if require_self_contained:
            try:
                resolved.relative_to(organized_root)
            except ValueError as exc:
                raise PreparedTrainingViewError(
                    "Prepared view is not self-contained."
                ) from exc
        if source_id in source_paths:
            raise PreparedTrainingViewError("Duplicate source_dataset_id.")
        source_paths[source_id] = resolved
        source_fingerprints[source_id] = str(
            value.get("source_dataset_fingerprint", "")
        )
    if tuple(manifest.get("source_dataset_ids", ())) != tuple(source_paths):
        raise PreparedTrainingViewError(
            "Manifest and source dataset identities differ."
        )
    return source_paths, source_fingerprints


def _load_tables(
    root: Path,
    derived: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    try:
        import pyarrow.parquet as pq

        episodes = pq.read_table(root / EPISODES_FILE)
        intervals = pq.read_table(root / INTERVALS_FILE)
        anchors = pq.read_table(root / ANCHORS_FILE)
    except Exception as exc:
        raise PreparedTrainingViewError(
            f"Cannot read prepared-view Parquet tables: {exc}"
        ) from exc
    expected_schemas = arrow_schemas()
    for filename, table, schema in (
        (EPISODES_FILE, episodes, expected_schemas[0]),
        (INTERVALS_FILE, intervals, expected_schemas[1]),
        (ANCHORS_FILE, anchors, expected_schemas[2]),
    ):
        if not table.schema.equals(schema, check_metadata=False):
            raise PreparedTrainingViewError(
                f"Prepared-view table schema mismatch: {filename}."
            )
        if table.num_rows != int(derived[filename].get("row_count", -1)):
            raise PreparedTrainingViewError(
                f"Prepared-view row count mismatch: {filename}."
            )
    if (
        episodes.num_rows != int(manifest.get("episode_count", -1))
        or intervals.num_rows != int(manifest.get("interval_count", -1))
        or anchors.num_rows != int(manifest.get("anchor_count", -1))
    ):
        raise PreparedTrainingViewError(
            "Prepared-view manifest summary counts are inconsistent."
        )
    return episodes, intervals, anchors


def _load_skill_projection(
    root: Path,
    derived: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    episodes: Any,
) -> tuple[Any | None, tuple[Mapping[str, Any], ...]]:
    capability = manifest.get("skill_projection")
    if capability is None:
        return None, ()
    if not isinstance(capability, Mapping):
        raise PreparedTrainingViewError(
            "Prepared-view Skill projection capability is invalid."
        )
    progress_hashes = sorted(
        set(
            str(value)
            for value in episodes["progress_profile_hash"].to_pylist()
        )
    )
    stage_hashes = sorted(
        set(
            str(value)
            for value in episodes["stage_profile_hash"].to_pylist()
            if value is not None
        )
    )
    if (
        progress_hashes
        != sorted(
            str(value)
            for value in manifest.get("progress_profile_hashes", ())
        )
        or stage_hashes
        != sorted(
            str(value)
            for value in manifest.get("stage_profile_hashes", ())
        )
        or manifest.get("progress_profile_hash")
        != (progress_hashes[0] if len(progress_hashes) == 1 else None)
        or manifest.get("stage_profile_hash")
        != (stage_hashes[0] if len(stage_hashes) == 1 else None)
    ):
        raise PreparedTrainingViewError(
            "Prepared Skill projection profile hashes and Episode rows differ."
        )
    try:
        skill_payload = capability.get("skill_contract")
        projection_payloads = capability.get("projections")
        if not isinstance(skill_payload, Mapping) or not isinstance(
            projection_payloads,
            (tuple, list),
        ):
            raise ValueError("Skill/projection definitions are missing.")
        skill = SkillTrainingContractV1.from_payload(skill_payload)
        projections = tuple(
            ProgressSkillProjectionV1.from_payload(item)
            for item in projection_payloads
        )
    except (TypeError, ValueError) as exc:
        raise PreparedTrainingViewError(
            f"Prepared Skill projection definitions are invalid: {exc}"
        ) from exc
    projection_hashes = {
        projection.semantic_content_hash for projection in projections
    }
    if (
        sorted(projection_hashes)
        != sorted(
            str(value)
            for value in capability.get("projection_hashes", ())
        )
        or any(
            projection.skill_training_contract_hash
            != skill.semantic_content_hash
            for projection in projections
        )
    ):
        raise PreparedTrainingViewError(
            "Prepared Skill projection definitions do not match the manifest."
        )
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(root / SKILL_INTERVALS_FILE)
    except Exception as exc:
        raise PreparedTrainingViewError(
            f"Cannot read prepared Skill interval table: {exc}"
        ) from exc
    if not table.schema.equals(
        skill_interval_arrow_schema(),
        check_metadata=False,
    ):
        raise PreparedTrainingViewError(
            "Prepared Skill interval table schema mismatch."
        )
    expected_count = int(
        derived[SKILL_INTERVALS_FILE].get("row_count", -1)
    )
    if (
        table.num_rows != expected_count
        or table.num_rows != int(capability.get("interval_count", -1))
    ):
        raise PreparedTrainingViewError(
            "Prepared Skill interval count is inconsistent."
        )
    records = read_cjsonl(
        root / SKILL_PROJECTION_RECEIPTS_FILE,
        domain=PREPARED_SKILL_PROJECTION_RECEIPTS_DOMAIN_V1,
    )
    if (
        len(records)
        != int(
            derived[SKILL_PROJECTION_RECEIPTS_FILE].get("row_count", -1)
        )
        or len(records) != int(capability.get("receipt_count", -1))
    ):
        raise PreparedTrainingViewError(
            "Prepared Skill projection receipt count is inconsistent."
        )
    expected_rows: list[dict[str, Any]] = []
    seen_ordinals: list[int] = []
    for record in records:
        ordinal = int(record.get("organized_episode_ordinal", -1))
        if ordinal < 0 or ordinal >= episodes.num_rows:
            raise PreparedTrainingViewError(
                "Prepared Skill projection receipt has an invalid Episode ordinal."
            )
        receipt_payload = record.get("receipt")
        if not isinstance(receipt_payload, Mapping):
            raise PreparedTrainingViewError(
                "Prepared Skill projection receipt record is invalid."
            )
        try:
            receipt = SkillProjectionReceiptV1.from_payload(receipt_payload)
        except (TypeError, ValueError) as exc:
            raise PreparedTrainingViewError(
                f"Prepared Skill projection receipt is invalid: {exc}"
            ) from exc
        if (
            receipt.skill_training_contract_hash != skill.semantic_content_hash
            or receipt.progress_skill_projection_hash not in projection_hashes
        ):
            raise PreparedTrainingViewError(
                "Prepared Skill projection receipt references a foreign "
                "Skill or projection."
            )
        expected_episode_hash = str(
            episodes["episode_capture_identity_hash"][ordinal].as_py()
        )
        if receipt.episode_capture_identity_hash != expected_episode_hash:
            raise PreparedTrainingViewError(
                "Prepared Skill projection receipt and Episode row differ."
            )
        frame_count = int(episodes["frame_count"][ordinal].as_py())
        if any(
            interval.sample_end > frame_count
            for interval in receipt.intervals
        ):
            raise PreparedTrainingViewError(
                "Prepared Skill projection interval exceeds its Episode axis."
            )
        seen_ordinals.append(ordinal)
        for interval in receipt.intervals:
            expected_rows.append(
                {
                    "organized_episode_ordinal": ordinal,
                    "episode_capture_identity_hash": expected_episode_hash,
                    "skill_id": skill.skill_id,
                    "skill_training_contract_hash": (
                        skill.semantic_content_hash
                    ),
                    "progress_skill_projection_hash": (
                        receipt.progress_skill_projection_hash
                    ),
                    "sample_begin": interval.sample_begin,
                    "sample_end_exclusive": interval.sample_end,
                    "training_domain": interval.training_domain,
                    "source_attempt_id": interval.source_attempt_id,
                    "source_stage_ids": list(interval.source_stage_ids),
                    "source_occurrence_ids": list(
                        interval.source_occurrence_ids
                    ),
                    "source_failure_fact_id": (
                        interval.source_failure_fact_id
                    ),
                }
            )
    if seen_ordinals != list(range(episodes.num_rows)):
        raise PreparedTrainingViewError(
            "Prepared Skill projection requires one ordered receipt per Episode."
        )
    if table.to_pylist() != expected_rows:
        raise PreparedTrainingViewError(
            "Prepared Skill interval table and projection receipts differ."
        )
    return table, records


def _verify_selection(episodes: Any, selected: tuple[Mapping[str, Any], ...]) -> None:
    if episodes.num_rows != len(selected):
        raise PreparedTrainingViewError(
            "Selected Episode rows and episodes.parquet differ."
        )
    identities = tuple(
        str(value) for value in episodes["episode_capture_identity_hash"].to_pylist()
    )
    if len(identities) != len(set(identities)):
        raise PreparedTrainingViewError("Prepared view has duplicate Episode identity.")
    table_selected = tuple(
        (
            int(episodes["organized_episode_ordinal"][index].as_py()),
            str(episodes["source_dataset_id"][index].as_py()),
            int(episodes["source_episode_index"][index].as_py()),
            str(episodes["episode_capture_identity_hash"][index].as_py()),
            str(episodes["source_episode_content_hash"][index].as_py()),
            episodes["annotation_set_id"][index].as_py(),
            episodes["annotation_manifest_hash"][index].as_py(),
            episodes["failure_trigger_hash"][index].as_py(),
        )
        for index in range(episodes.num_rows)
    )
    declared_selected = tuple(
        (
            int(record.get("organized_episode_ordinal", -1)),
            str(record.get("source_dataset_id", "")),
            int(record.get("source_episode_index", -1)),
            str(record.get("episode_capture_identity_hash", "")),
            str(record.get("source_episode_content_hash", "")),
            record.get("annotation_set_id"),
            record.get("annotation_manifest_hash"),
            record.get("failure_trigger_hash"),
        )
        for record in selected
    )
    if table_selected != declared_selected:
        raise PreparedTrainingViewError(
            "selected_episodes.cjsonl and episodes.parquet differ."
        )


def _validate_inventory_locators(
    inventory: tuple[Mapping[str, Any], ...],
    source_paths: Mapping[str, Path],
) -> None:
    for record in inventory:
        source_id = str(record.get("source_dataset_id", ""))
        relative = str(record.get("relative_path", ""))
        pure = PurePosixPath(relative)
        if source_id not in source_paths or not relative or pure.is_absolute():
            raise PreparedTrainingViewError(
                "Source inventory contains an invalid locator."
            )
        candidate = (source_paths[source_id] / Path(*pure.parts)).resolve()
        try:
            candidate.relative_to(source_paths[source_id])
        except ValueError as exc:
            raise PreparedTrainingViewError(
                "Inventory path escapes source root."
            ) from exc


def _verify_sources(
    inventory: tuple[Mapping[str, Any], ...],
    source_paths: Mapping[str, Path],
    source_fingerprints: Mapping[str, str],
) -> None:
    for source_id, source_root in source_paths.items():
        if (
            not source_root.is_dir()
            or source_dataset_base_fingerprint(source_root)
            != source_fingerprints[source_id]
        ):
            raise PreparedTrainingViewError(
                f"Source dataset fingerprint mismatch: {source_id}."
            )
    for record in inventory:
        source_id = str(record.get("source_dataset_id", ""))
        relative = str(record.get("relative_path", ""))
        source_root = source_paths[source_id]
        pure = PurePosixPath(relative)
        path = (source_root / Path(*pure.parts)).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(path) != str(record.get("sha256", ""))
        ):
            raise PreparedTrainingViewError(
                f"Source inventory verification failed: {source_id}:{relative}."
            )


def load_prepared_training_view(
    view_root: str | Path,
    *,
    organized_dataset_root: str | Path | None = None,
    verify_source_files: bool = False,
    require_self_contained: bool = False,
) -> LoadedPreparedTrainingView:
    """Strictly load a complete view; optionally verify every source hash."""

    root = Path(view_root).expanduser().resolve()
    if not root.is_dir():
        raise PreparedTrainingViewError(f"Prepared view does not exist: {root}.")
    organized_root = (
        root.parents[2]
        if organized_dataset_root is None
        else Path(organized_dataset_root).expanduser().resolve()
    )
    manifest = _load_manifest(root)
    derived = _verify_derived_files(root, manifest)
    source_paths, source_fingerprints = _load_sources(
        root=root,
        organized_root=organized_root,
        manifest=manifest,
        require_self_contained=require_self_contained,
    )
    selected = read_cjsonl(
        root / SELECTED_EPISODES_FILE,
        domain=PREPARED_SELECTED_EPISODE_DOMAIN_V1,
    )
    inventory = read_cjsonl(
        root / SOURCE_INVENTORY_FILE,
        domain=PREPARED_SOURCE_INVENTORY_DOMAIN_V1,
    )
    episodes, intervals, anchors = _load_tables(root, derived, manifest)
    skill_intervals, skill_receipts = _load_skill_projection(
        root,
        derived,
        manifest,
        episodes,
    )
    _verify_selection(episodes, selected)
    _validate_inventory_locators(inventory, source_paths)
    if verify_source_files:
        _verify_sources(inventory, source_paths, source_fingerprints)
    return LoadedPreparedTrainingView(
        view_root=root,
        organized_dataset_root=organized_root,
        training_view_hash=str(manifest["training_view_hash"]),
        manifest=manifest,
        source_datasets=source_paths,
        selected_episodes=selected,
        source_inventory=inventory,
        episodes=episodes,
        semantic_intervals=intervals,
        semantic_anchors=anchors,
        skill_intervals=skill_intervals,
        skill_projection_receipts=skill_receipts,
    )


__all__ = ["load_prepared_training_view"]
