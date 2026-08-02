"""Read-only loading and verification for sealed progress Episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_facts import (
    PROGRESS_BINDING_FILE_DOMAIN_V1,
    PROGRESS_FACT_FILE_DOMAIN_V1,
    PROGRESS_MANIFEST_FILE_DOMAIN_V1,
    PROGRESS_PROFILE_FILE_DOMAIN_V1,
    ProgressEpisodeManifestV1,
    ProgressFactEnvelopeV1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stages import (
    PROGRESS_STAGE_ENTRY_BINDING_FILE_DOMAIN_V1,
    TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
    ProgressStageEntryBindingV1,
    TaskStageProfileV1,
    validate_progress_stage_entry_binding,
    validate_task_stage_profile,
)

from .progress_timeline import (
    ResolvedProgressTimeline,
    resolve_progress_timeline,
)


class ProgressArtifactReadError(ValueError):
    """Raised when a persisted progress Episode is absent or inconsistent."""


@dataclass(frozen=True)
class SealedProgressEpisode:
    """Verified read view over one immutable progress Episode."""

    episode_dir: Path
    profile: TaskProgressProfileV1
    binding: ProgressEpisodeBindingV1
    facts: tuple[ProgressFactEnvelopeV1, ...]
    manifest: ProgressEpisodeManifestV1
    timeline: ResolvedProgressTimeline
    stage_profile: TaskStageProfileV1 | None = None
    stage_entry_binding: ProgressStageEntryBindingV1 | None = None


DATASET_PROGRESS_PROFILE_RELATIVE_PATH = Path("meta/progress/profile.cjson")
DATASET_TASK_STAGE_PROFILE_RELATIVE_PATH = Path("meta/progress/stages.cjson")
DATASET_PROGRESS_EPISODES_RELATIVE_ROOT = Path("meta/progress/episodes")


def _read_canonical_mapping(path: Path, *, domain: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ProgressArtifactReadError(f"Missing progress artifact: {path.name}.")
    try:
        payload = canonical_loads(path.read_bytes(), expected_domain=domain)
    except (OSError, TypeError, ValueError) as exc:
        raise ProgressArtifactReadError(
            f"Cannot read progress artifact {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProgressArtifactReadError(
            f"Progress artifact {path.name} must contain an object."
        )
    return payload


def _read_facts(path: Path) -> tuple[ProgressFactEnvelopeV1, ...]:
    if not path.is_file():
        raise ProgressArtifactReadError("Missing progress artifact: facts.cjsonl.")
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise ProgressArtifactReadError(
            f"Cannot read progress artifact facts.cjsonl: {exc}"
        ) from exc
    if not encoded:
        raise ProgressArtifactReadError("Progress fact log is empty.")
    if not encoded.endswith(b"\n"):
        raise ProgressArtifactReadError(
            "Progress fact log has a torn final record."
        )
    facts: list[ProgressFactEnvelopeV1] = []
    for line_number, line in enumerate(encoded.splitlines(), start=1):
        if not line:
            raise ProgressArtifactReadError(
                f"Progress fact log contains an empty line at {line_number}."
            )
        try:
            payload = canonical_loads(
                line,
                expected_domain=PROGRESS_FACT_FILE_DOMAIN_V1,
            )
            if not isinstance(payload, Mapping):
                raise TypeError("fact payload is not an object")
            facts.append(ProgressFactEnvelopeV1.from_payload(payload))
        except (TypeError, ValueError) as exc:
            raise ProgressArtifactReadError(
                f"Invalid progress fact at line {line_number}: {exc}"
            ) from exc
    return tuple(facts)


def _read_profile(path: Path) -> TaskProgressProfileV1:
    try:
        return TaskProgressProfileV1.from_payload(
            _read_canonical_mapping(
                path,
                domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
            )
        )
    except ProgressArtifactReadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProgressArtifactReadError(
            f"Invalid progress profile {path}: {exc}"
        ) from exc


def _read_stage_profile(path: Path) -> TaskStageProfileV1:
    try:
        return TaskStageProfileV1.from_payload(
            _read_canonical_mapping(
                path,
                domain=TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
            )
        )
    except ProgressArtifactReadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProgressArtifactReadError(
            f"Invalid task stage profile {path}: {exc}"
        ) from exc


def _require_expected_hash(
    *,
    label: str,
    expected: str | None,
    actual: str,
) -> None:
    if expected is None:
        return
    expected_value = str(expected).strip()
    if not expected_value:
        raise ProgressArtifactReadError(f"Expected {label} must not be empty.")
    if expected_value != actual:
        raise ProgressArtifactReadError(
            f"Progress {label} mismatch: expected={expected_value} actual={actual}."
        )


def load_dataset_progress_profile(
    dataset_root: str | Path,
    *,
    require_explicit_binding: bool = False,
) -> TaskProgressProfileV1 | None:
    """Verify the single progress profile used throughout one dataset.

    The reader is intentionally independent of ``collection_runtime`` and
    never backfills the root binding. Legacy Episode-local profiles are
    accepted only when they are all exactly identical.
    """

    root = Path(dataset_root).expanduser().resolve()
    binding_path = root / DATASET_PROGRESS_PROFILE_RELATIVE_PATH
    explicit = _read_profile(binding_path) if binding_path.is_file() else None
    episode_root = root / DATASET_PROGRESS_EPISODES_RELATIVE_ROOT
    episode_profiles: list[TaskProgressProfileV1] = []
    if episode_root.exists():
        for episode_dir in sorted(
            (path for path in episode_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            episode_profiles.append(
                _read_profile(episode_dir / "profile.cjson")
            )
    expected = explicit or (
        episode_profiles[0] if episode_profiles else None
    )
    if expected is None:
        if require_explicit_binding:
            raise ProgressArtifactReadError(
                "Dataset has no explicit progress profile binding."
            )
        return None
    mismatched = tuple(
        profile.semantic_content_hash
        for profile in episode_profiles
        if profile.semantic_content_hash != expected.semantic_content_hash
    )
    if mismatched:
        raise ProgressArtifactReadError(
            "Dataset contains progress Episodes with different profile hashes."
        )
    if explicit is None and require_explicit_binding:
        raise ProgressArtifactReadError(
            "Legacy progress Episodes are consistent, but the dataset has no "
            "explicit progress profile binding."
        )
    return expected


def load_dataset_task_stage_profile(
    dataset_root: str | Path,
    *,
    progress_profile: TaskProgressProfileV1,
) -> TaskStageProfileV1 | None:
    """Read the exact optional fine-stage profile shared by all Episodes."""

    root = Path(dataset_root).expanduser().resolve()
    explicit_path = root / DATASET_TASK_STAGE_PROFILE_RELATIVE_PATH
    explicit = (
        _read_stage_profile(explicit_path)
        if explicit_path.is_file()
        else None
    )
    episode_root = root / DATASET_PROGRESS_EPISODES_RELATIVE_ROOT
    episode_profiles: list[TaskStageProfileV1] = []
    missing = 0
    if episode_root.exists():
        for episode_dir in sorted(
            (path for path in episode_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            stage_path = episode_dir / "stages.cjson"
            if stage_path.is_file():
                episode_profiles.append(_read_stage_profile(stage_path))
            else:
                missing += 1
    if episode_profiles and missing:
        raise ProgressArtifactReadError(
            "Dataset mixes stage-profiled and unprofiled progress Episodes."
        )
    expected = explicit or (
        episode_profiles[0] if episode_profiles else None
    )
    if expected is None:
        return None
    validate_task_stage_profile(
        progress_profile=progress_profile,
        stage_profile=expected,
    )
    if any(
        item.semantic_content_hash != expected.semantic_content_hash
        for item in episode_profiles
    ):
        raise ProgressArtifactReadError(
            "Dataset contains multiple task stage profile hashes."
        )
    return expected


def load_sealed_progress_episode(
    episode_dir: str | Path,
    *,
    expected_episode_identity_hash: str | None = None,
    expected_profile_hash: str | None = None,
    expected_binding_hash: str | None = None,
    expected_manifest_hash: str | None = None,
) -> SealedProgressEpisode:
    """Load one sealed Episode without creating files or opening write handles."""

    resolved_dir = Path(episode_dir).expanduser().resolve()
    if not resolved_dir.is_dir():
        raise ProgressArtifactReadError(
            f"Progress Episode directory does not exist: {resolved_dir}."
        )
    try:
        profile = _read_profile(resolved_dir / "profile.cjson")
        binding = ProgressEpisodeBindingV1.from_payload(
            _read_canonical_mapping(
                resolved_dir / "binding.cjson",
                domain=PROGRESS_BINDING_FILE_DOMAIN_V1,
            )
        )
        facts = _read_facts(resolved_dir / "facts.cjsonl")
        manifest = ProgressEpisodeManifestV1.from_payload(
            _read_canonical_mapping(
                resolved_dir / "manifest.cjson",
                domain=PROGRESS_MANIFEST_FILE_DOMAIN_V1,
            )
        )
        timeline = resolve_progress_timeline(
            profile=profile,
            binding=binding,
            facts=facts,
            manifest=manifest,
            require_sealed=True,
        )
        stage_path = resolved_dir / "stages.cjson"
        entry_path = resolved_dir / "stage_entry.cjson"
        if stage_path.exists() != entry_path.exists():
            raise ProgressArtifactReadError(
                "Fine-stage progress artifacts are incomplete."
            )
        stage_profile = (
            None if not stage_path.exists() else _read_stage_profile(stage_path)
        )
        stage_entry_binding = (
            None
            if not entry_path.exists()
            else ProgressStageEntryBindingV1.from_payload(
                _read_canonical_mapping(
                    entry_path,
                    domain=(
                        PROGRESS_STAGE_ENTRY_BINDING_FILE_DOMAIN_V1
                    ),
                )
            )
        )
        if stage_profile is not None and stage_entry_binding is not None:
            validate_progress_stage_entry_binding(
                progress_profile=profile,
                progress_entry_step_id=binding.entry_step_id,
                stage_profile=stage_profile,
                stage_binding=stage_entry_binding,
            )
    except ProgressArtifactReadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProgressArtifactReadError(
            f"Invalid sealed progress Episode: {exc}"
        ) from exc

    _require_expected_hash(
        label="Episode identity hash",
        expected=expected_episode_identity_hash,
        actual=binding.episode_capture_identity_hash,
    )
    _require_expected_hash(
        label="profile hash",
        expected=expected_profile_hash,
        actual=profile.semantic_content_hash,
    )
    _require_expected_hash(
        label="binding hash",
        expected=expected_binding_hash,
        actual=binding.semantic_content_hash,
    )
    _require_expected_hash(
        label="manifest hash",
        expected=expected_manifest_hash,
        actual=manifest.semantic_content_hash,
    )
    return SealedProgressEpisode(
        episode_dir=resolved_dir,
        profile=profile,
        binding=binding,
        facts=facts,
        manifest=manifest,
        timeline=timeline,
        stage_profile=stage_profile,
        stage_entry_binding=stage_entry_binding,
    )


__all__ = [
    "DATASET_PROGRESS_PROFILE_RELATIVE_PATH",
    "DATASET_TASK_STAGE_PROFILE_RELATIVE_PATH",
    "ProgressArtifactReadError",
    "SealedProgressEpisode",
    "load_dataset_progress_profile",
    "load_dataset_task_stage_profile",
    "load_sealed_progress_episode",
]
