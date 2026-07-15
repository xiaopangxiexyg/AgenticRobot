from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ARM_CONTROL_MODE_IMPEDANCE = "impedance"
ARM_CONTROL_MODE_POSITION = "position"
ARM_CONTROL_MODES = frozenset(
    {ARM_CONTROL_MODE_IMPEDANCE, ARM_CONTROL_MODE_POSITION}
)


class DatasetArmControlModeError(ValueError):
    """Raised when dataset execution-mode metadata is invalid or incompatible."""


@dataclass(frozen=True)
class DatasetArmControlModeContract:
    recorded_mode: str | None
    selected_mode: str
    source: str
    compatible: bool
    legacy_default_applied: bool = False
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_mode": self.recorded_mode,
            "selected_mode": self.selected_mode,
            "source": self.source,
            "compatible": bool(self.compatible),
            "legacy_default_applied": bool(self.legacy_default_applied),
            "warning": self.warning,
        }


def normalize_arm_control_mode(value: Any, *, label: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in ARM_CONTROL_MODES:
        raise DatasetArmControlModeError(
            f"{label} must be 'impedance' or 'position', got {value!r}."
        )
    return mode


def _episode_metadata(metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    direct = metadata.get("episode_metadata")
    if isinstance(direct, Mapping):
        return direct, "episode_metadata"
    source = metadata.get("source_metadata")
    if isinstance(source, Mapping):
        nested = source.get("episode_metadata")
        if isinstance(nested, Mapping):
            return nested, "source_metadata.episode_metadata"
    return metadata, "episode_metadata"


def _recorded_mode_candidates(
    metadata: Mapping[str, Any],
) -> tuple[list[tuple[str, str]], str]:
    episode, prefix = _episode_metadata(metadata)
    candidates: list[tuple[str, str]] = []

    raw_mode = episode.get("arm_control_mode")
    if raw_mode not in (None, ""):
        candidates.append(
            (
                normalize_arm_control_mode(
                    raw_mode,
                    label=f"{prefix}.arm_control_mode",
                ),
                f"{prefix}.arm_control_mode",
            )
        )

    profile = episode.get("arm_execution_profile")
    if isinstance(profile, Mapping) and profile.get("mode") not in (None, ""):
        candidates.append(
            (
                normalize_arm_control_mode(
                    profile.get("mode"),
                    label=f"{prefix}.arm_execution_profile.mode",
                ),
                f"{prefix}.arm_execution_profile.mode",
            )
        )

    calibration = episode.get("calibration_snapshot")
    calibration_metadata = (
        calibration.get("metadata") if isinstance(calibration, Mapping) else None
    )
    if isinstance(calibration_metadata, Mapping):
        calibration_mode = calibration_metadata.get("arm_control_mode")
        if calibration_mode not in (None, ""):
            candidates.append(
                (
                    normalize_arm_control_mode(
                        calibration_mode,
                        label=(
                            f"{prefix}.calibration_snapshot.metadata.arm_control_mode"
                        ),
                    ),
                    f"{prefix}.calibration_snapshot.metadata.arm_control_mode",
                )
            )
    return candidates, prefix


def validate_dataset_arm_control_mode(
    metadata: Mapping[str, Any] | None,
    *,
    selected_mode: Any,
    hardware_output: bool,
) -> DatasetArmControlModeContract:
    """Validate the dataset's recorded arm branch against the selected branch.

    Legacy episodes without this contract remain inspectable in dry-run mode. For
    real hardware output they are deliberately treated as impedance recordings,
    so an old episode can never silently enter the default position branch.
    """

    selected = normalize_arm_control_mode(selected_mode, label="selected arm control mode")
    payload = metadata if isinstance(metadata, Mapping) else {}
    candidates, prefix = _recorded_mode_candidates(payload)
    distinct = sorted({mode for mode, _source in candidates})
    if len(distinct) > 1:
        details = ", ".join(f"{source}={mode}" for mode, source in candidates)
        raise DatasetArmControlModeError(
            "Dataset arm-control metadata is contradictory: " + details
        )

    legacy_default_applied = False
    warning: str | None = None
    if distinct:
        recorded = distinct[0]
        source = next(source for mode, source in candidates if mode == recorded)
    elif hardware_output:
        recorded = ARM_CONTROL_MODE_IMPEDANCE
        source = f"{prefix}.arm_control_mode:legacy_hardware_default"
        legacy_default_applied = True
        warning = (
            "Dataset episode has no arm_control_mode; real hardware output treats "
            "this legacy episode as impedance."
        )
    else:
        recorded = None
        source = f"{prefix}.arm_control_mode:missing"
        warning = (
            "Dataset episode has no arm_control_mode; dry-run inspection cannot "
            "verify the selected execution branch."
        )

    compatible = recorded is None or recorded == selected
    contract = DatasetArmControlModeContract(
        recorded_mode=recorded,
        selected_mode=selected,
        source=source,
        compatible=compatible,
        legacy_default_applied=legacy_default_applied,
        warning=warning,
    )
    if not compatible:
        legacy = " (legacy missing-mode default)" if legacy_default_applied else ""
        raise DatasetArmControlModeError(
            "Dataset arm_control_mode mismatch: "
            f"recorded={recorded!r}{legacy}, selected={selected!r}, source={source}. "
            "Select the matching replay/deploy execution mode before hardware output."
        )
    return contract


__all__ = [
    "ARM_CONTROL_MODE_IMPEDANCE",
    "ARM_CONTROL_MODE_POSITION",
    "DatasetArmControlModeContract",
    "DatasetArmControlModeError",
    "normalize_arm_control_mode",
    "validate_dataset_arm_control_mode",
]
