"""Read-only loading, consistency, and future-merge assessment for v2.5-p."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.control_trace_episode import EPISODE_COMMIT_FILE_DOMAIN_V1
from dataset_contracts.embodiment import (
    CaptureProjectionProfile,
    EmbodimentCaptureProfile,
    EpisodeEmbodimentBinding,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.task_semantics import (
    LongHorizonTaskProfile,
    SubtaskAttemptEvent,
    TaskSemanticsEpisodeManifest,
)
from dataset_contracts.v25p import (
    V25P_DATASET_MANIFEST_FILE_DOMAIN_V1,
    V25P_EMBODIMENT_BINDING_FILE_DOMAIN_V1,
    V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
    V25P_EPISODE_MANIFEST_FILE_DOMAIN_V1,
    V25P_EPISODE_RELATIVE_ROOT_V1,
    V25P_PROFILE_FILE_DOMAIN_V1,
    V25P_PROFILE_RELATIVE_ROOT_V1,
    V25P_RELATIVE_ROOT_V1,
    V25P_TASK_EVENT_FILE_DOMAIN_V1,
    V25P_TASK_MANIFEST_FILE_DOMAIN_V1,
    V25pDatasetManifest,
    V25pEpisodeManifest,
    v25p_base_episode_manifest_hash,
)


V25P_ASSESSMENT_INVALID = "invalid"
V25P_ASSESSMENT_V23_PROVISIONAL = "v23_provisional_complete"
V25P_ASSESSMENT_V25_COMPATIBILITY = "v25_compatibility_candidate"
V25P_ASSESSMENT_V25_NATIVE_MERGE = "v25_native_merge_candidate"


@dataclass(frozen=True)
class V25pMergeReadiness:
    classification: str
    valid: bool
    native_v25: bool
    reasons: tuple[str, ...]


class V25pReadError(ValueError):
    """Raised when a persisted provisional capability is absent or inconsistent."""


@dataclass(frozen=True)
class V25pLoadedEpisode:
    dataset_manifest: V25pDatasetManifest
    episode_manifest: V25pEpisodeManifest
    identity: EpisodeCaptureIdentity
    embodiment_profile: EmbodimentCaptureProfile
    projection_profile: CaptureProjectionProfile
    embodiment_binding: EpisodeEmbodimentBinding
    task_profile: LongHorizonTaskProfile | None
    task_events: tuple[SubtaskAttemptEvent, ...]
    task_semantics_manifest: TaskSemanticsEpisodeManifest | None
    base_episode_record: Mapping[str, Any]
    readiness: V25pMergeReadiness


def _canonical_mapping(path: Path, *, domain: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise V25pReadError(f"Required v2.5-p artifact is missing: {path}")
    try:
        payload = canonical_loads(path.read_bytes(), expected_domain=domain)
    except (OSError, TypeError, ValueError) as exc:
        raise V25pReadError(f"Invalid v2.5-p artifact {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise V25pReadError(f"v2.5-p artifact is not a mapping: {path}")
    return payload


def _legacy_episode_record(dataset_root: Path, episode_index: int) -> Mapping[str, Any]:
    path = dataset_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise V25pReadError(f"Legacy episode manifest is missing: {path}")
    matches: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise V25pReadError(
                        f"Legacy episode record is not an object at {path}:{line_number}"
                    )
                if int(payload.get("episode_index", -1)) == episode_index:
                    matches.append(dict(payload))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, V25pReadError):
            raise
        raise V25pReadError(f"Cannot read legacy episode manifest {path}: {exc}") from exc
    if len(matches) != 1:
        raise V25pReadError(
            "Expected exactly one finalized legacy episode record for "
            f"episode_index={episode_index}; found {len(matches)}."
        )
    return matches[0]


def load_v25p_episode(
    dataset_root: str | Path,
    episode_uuid: str,
) -> V25pLoadedEpisode:
    """Load and verify one sealed v2.5-p episode without changing the dataset."""

    root = Path(dataset_root).expanduser().resolve()
    uuid_text = str(episode_uuid).strip()
    if not uuid_text or uuid_text in {".", ".."} or "/" in uuid_text or "\\" in uuid_text:
        raise V25pReadError("episode_uuid is empty or unsafe.")
    dataset_manifest = V25pDatasetManifest.from_payload(
        _canonical_mapping(
            root / V25P_RELATIVE_ROOT_V1 / "dataset_manifest.cjson",
            domain=V25P_DATASET_MANIFEST_FILE_DOMAIN_V1,
        )
    )
    episode_dir = root / V25P_EPISODE_RELATIVE_ROOT_V1 / uuid_text
    identity = EpisodeCaptureIdentity.from_payload(
        _canonical_mapping(
            episode_dir / "identity.cjson",
            domain=V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
        )
    )
    if identity.episode_uuid != uuid_text:
        raise V25pReadError("Episode directory and identity UUID do not match.")
    binding = EpisodeEmbodimentBinding.from_payload(
        _canonical_mapping(
            episode_dir / "embodiment_binding.cjson",
            domain=V25P_EMBODIMENT_BINDING_FILE_DOMAIN_V1,
        )
    )
    embodiment_profile = EmbodimentCaptureProfile.from_payload(
        _canonical_mapping(
            root
            / V25P_PROFILE_RELATIVE_ROOT_V1
            / "embodiment"
            / f"{binding.embodiment_profile_hash}.cjson",
            domain=V25P_PROFILE_FILE_DOMAIN_V1,
        )
    )
    projection_profile = CaptureProjectionProfile.from_payload(
        _canonical_mapping(
            root
            / V25P_PROFILE_RELATIVE_ROOT_V1
            / "capture_projection"
            / f"{binding.capture_projection_profile_hash}.cjson",
            domain=V25P_PROFILE_FILE_DOMAIN_V1,
        )
    )
    episode_manifest = V25pEpisodeManifest.from_payload(
        _canonical_mapping(
            episode_dir / "episode_manifest.cjson",
            domain=V25P_EPISODE_MANIFEST_FILE_DOMAIN_V1,
        )
    )

    task_manifest: TaskSemanticsEpisodeManifest | None = None
    task_profile: LongHorizonTaskProfile | None = None
    task_events: tuple[SubtaskAttemptEvent, ...] = ()
    if episode_manifest.task_semantics_manifest_hash is not None:
        task_manifest = TaskSemanticsEpisodeManifest.from_payload(
            _canonical_mapping(
                episode_dir / "task_manifest.cjson",
                domain=V25P_TASK_MANIFEST_FILE_DOMAIN_V1,
            )
        )
        task_profile = LongHorizonTaskProfile.from_payload(
            _canonical_mapping(
                root
                / V25P_PROFILE_RELATIVE_ROOT_V1
                / "task"
                / f"{task_manifest.task_profile_hash}.cjson",
                domain=V25P_PROFILE_FILE_DOMAIN_V1,
            )
        )
        event_paths = sorted((episode_dir / "task_events").glob("*.cjson"))
        task_events = tuple(
            SubtaskAttemptEvent.from_payload(
                _canonical_mapping(path, domain=V25P_TASK_EVENT_FILE_DOMAIN_V1)
            )
            for path in event_paths
        )
        rebuilt = TaskSemanticsEpisodeManifest.create(
            profile=task_profile,
            episode_uuid=identity.episode_uuid,
            legacy_episode_index=identity.legacy_episode_index,
            episode_identity_hash=identity.semantic_content_hash,
            events=task_events,
            sealed_at=task_manifest.sealed_at,
            sealed_clock_domain_id=task_manifest.sealed_clock_domain_id,
        )
        if rebuilt.semantic_content_hash != task_manifest.semantic_content_hash:
            raise V25pReadError(
                "Task event files do not reproduce the sealed task manifest."
            )
    elif (episode_dir / "task_manifest.cjson").exists() or (
        episode_dir / "task_events"
    ).exists():
        raise V25pReadError(
            "Task artifacts exist but the sealed episode does not declare task semantics."
        )

    base_record = _legacy_episode_record(root, identity.legacy_episode_index)
    if (
        v25p_base_episode_manifest_hash(base_record)
        != episode_manifest.base_episode_manifest_hash
    ):
        raise V25pReadError(
            "The finalized legacy episode record no longer matches the v2.5-p seal."
        )
    if episode_manifest.v24_capture_commit_hash is not None:
        commit = _canonical_mapping(
            root
            / "meta"
            / "v24"
            / "control_trace"
            / "episodes"
            / identity.episode_uuid
            / "capture_commit.cjson",
            domain=EPISODE_COMMIT_FILE_DOMAIN_V1,
        )
        if str(commit.get("semantic_content_hash", "")) != (
            episode_manifest.v24_capture_commit_hash
        ):
            raise V25pReadError("Referenced v2.4 capture commit hash does not match.")
        if (
            str(commit.get("episode_uuid", "")),
            int(commit.get("legacy_episode_index", -1)),
        ) != (identity.episode_uuid, identity.legacy_episode_index):
            raise V25pReadError("Referenced v2.4 capture commit identity does not match.")

    readiness = assess_v25p_merge_readiness(
        dataset_manifest=dataset_manifest,
        episode_manifest=episode_manifest,
        identity=identity,
        embodiment_profile=embodiment_profile,
        projection_profile=projection_profile,
        embodiment_binding=binding,
        task_semantics_manifest=task_manifest,
    )
    if not readiness.valid:
        raise V25pReadError("; ".join(readiness.reasons))
    return V25pLoadedEpisode(
        dataset_manifest=dataset_manifest,
        episode_manifest=episode_manifest,
        identity=identity,
        embodiment_profile=embodiment_profile,
        projection_profile=projection_profile,
        embodiment_binding=binding,
        task_profile=task_profile,
        task_events=task_events,
        task_semantics_manifest=task_manifest,
        base_episode_record=base_record,
        readiness=readiness,
    )


def assess_v25p_merge_readiness(
    *,
    dataset_manifest: V25pDatasetManifest,
    episode_manifest: V25pEpisodeManifest,
    identity: EpisodeCaptureIdentity,
    embodiment_profile: EmbodimentCaptureProfile,
    projection_profile: CaptureProjectionProfile,
    embodiment_binding: EpisodeEmbodimentBinding,
    task_semantics_manifest: TaskSemanticsEpisodeManifest | None = None,
) -> V25pMergeReadiness:
    """Check bindings and classify a provisional episode without mutating it."""

    reasons: list[str] = []
    episode_pair = (identity.episode_uuid, identity.legacy_episode_index)
    for label, pair in (
        (
            "episode_manifest",
            (episode_manifest.episode_uuid, episode_manifest.legacy_episode_index),
        ),
        (
            "embodiment_binding",
            (embodiment_binding.episode_uuid, embodiment_binding.legacy_episode_index),
        ),
    ):
        if pair != episode_pair:
            reasons.append(f"{label} episode identity mismatch")
    if embodiment_binding.episode_identity_hash != identity.semantic_content_hash:
        reasons.append("embodiment binding identity hash mismatch")
    if episode_manifest.episode_identity_hash != identity.semantic_content_hash:
        reasons.append("episode manifest identity hash mismatch")
    if embodiment_binding.embodiment_profile_hash != embodiment_profile.semantic_content_hash:
        reasons.append("embodiment profile hash mismatch")
    if (
        embodiment_binding.capture_projection_profile_hash
        != projection_profile.semantic_content_hash
    ):
        reasons.append("capture projection profile hash mismatch")
    if embodiment_profile.semantic_content_hash not in dataset_manifest.embodiment_profile_hashes:
        reasons.append("embodiment profile is not allowed by the dataset manifest")
    if (
        projection_profile.semantic_content_hash
        not in dataset_manifest.capture_projection_profile_hashes
    ):
        reasons.append("capture projection is not allowed by the dataset manifest")
    if projection_profile.policy_io_layout_hash != dataset_manifest.policy_io_layout_hash:
        reasons.append("policy I/O layout hash mismatch")
    if projection_profile.pose_frame_contract_hash != dataset_manifest.pose_frame_contract_hash:
        reasons.append("pose-frame contract hash mismatch")
    if episode_manifest.embodiment_binding_hash != embodiment_binding.semantic_content_hash:
        reasons.append("episode manifest embodiment binding hash mismatch")

    embodiment_components = {
        component.component_id: component for component in embodiment_profile.components
    }
    projected_roles: dict[str, set[str]] = {}
    for projection in projection_profile.components:
        component = embodiment_components.get(projection.component_id)
        if component is None:
            reasons.append(
                f"projection references unknown component {projection.component_id!r}"
            )
            continue
        projected_roles.setdefault(projection.component_id, set()).add(
            projection.dataset_role
        )
        expected_hashes: tuple[str | None, ...]
        if projection.dataset_role == "action":
            expected_hashes = (component.command_payload_schema_hash,)
        elif projection.dataset_role == "state":
            expected_hashes = (component.measurement_payload_schema_hash,)
        else:
            expected_hashes = (
                component.command_payload_schema_hash,
                component.measurement_payload_schema_hash,
            )
        if projection.value_schema_hash not in {
            value for value in expected_hashes if value is not None
        }:
            reasons.append(
                f"projection schema does not match component {projection.component_id!r} "
                f"role {projection.dataset_role!r}"
            )
    for component in embodiment_profile.components:
        if not component.required_for_capture:
            continue
        roles = projected_roles.get(component.component_id, set())
        if component.command_payload_schema_hash is not None and "action" not in roles:
            reasons.append(
                f"required component {component.component_id!r} has no action projection"
            )
        if component.measurement_payload_schema_hash is not None and "state" not in roles:
            reasons.append(
                f"required component {component.component_id!r} has no state projection"
            )

    declared_task_hash = episode_manifest.task_semantics_manifest_hash
    if declared_task_hash is None and task_semantics_manifest is not None:
        reasons.append("task manifest supplied but episode capability is absent")
    elif declared_task_hash is not None and task_semantics_manifest is None:
        reasons.append("episode declares task semantics but its manifest is missing")
    elif task_semantics_manifest is not None:
        if declared_task_hash != task_semantics_manifest.semantic_content_hash:
            reasons.append("task semantics manifest hash mismatch")
        if (
            task_semantics_manifest.episode_uuid,
            task_semantics_manifest.legacy_episode_index,
        ) != episode_pair:
            reasons.append("task semantics episode identity mismatch")
        if task_semantics_manifest.episode_identity_hash != identity.semantic_content_hash:
            reasons.append("task semantics identity hash mismatch")
        if (
            task_semantics_manifest.task_profile_hash
            not in dataset_manifest.long_horizon_task_profile_hashes
        ):
            reasons.append("task profile is not allowed by the dataset manifest")

    if reasons:
        return V25pMergeReadiness(
            classification=V25P_ASSESSMENT_INVALID,
            valid=False,
            native_v25=False,
            reasons=tuple(reasons),
        )
    if episode_manifest.v24_capture_commit_hash is not None:
        return V25pMergeReadiness(
            classification=V25P_ASSESSMENT_V25_NATIVE_MERGE,
            valid=True,
            native_v25=False,
            reasons=(
                "All provisional bindings are consistent and a v2.4 capture commit "
                "is referenced; a future authoritative v2.5 commit is still required.",
            ),
        )
    classification = (
        V25P_ASSESSMENT_V25_COMPATIBILITY
        if task_semantics_manifest is not None
        else V25P_ASSESSMENT_V23_PROVISIONAL
    )
    return V25pMergeReadiness(
        classification=classification,
        valid=True,
        native_v25=False,
        reasons=(
            "Bindings are internally consistent; without a complete v2.4 capture "
            "commit this remains a compatibility/provisional view.",
        ),
    )


__all__ = [
    "V25P_ASSESSMENT_INVALID",
    "V25P_ASSESSMENT_V23_PROVISIONAL",
    "V25P_ASSESSMENT_V25_COMPATIBILITY",
    "V25P_ASSESSMENT_V25_NATIVE_MERGE",
    "V25pMergeReadiness",
    "V25pLoadedEpisode",
    "V25pReadError",
    "assess_v25p_merge_readiness",
    "load_v25p_episode",
]
