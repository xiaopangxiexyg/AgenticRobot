"""Immutable dataset and per-sample contracts for exclusive HITL capture.

The runtime may switch between a human teleoperator and one policy source, but
the persisted action has exactly one causal owner.  These contracts deliberately
describe that executed-source lineage; they do not reconstruct counterfactual
policy actions while the human owns control and they do not authorize RTC
motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


EXCLUSIVE_HITL_DATASET_SCHEMA_V1 = "dexteleop.exclusive_hitl_capture.v1"
EXCLUSIVE_HITL_SAMPLE_SCHEMA_V1 = "dexteleop.exclusive_hitl_sample.v1"
EXCLUSIVE_HITL_DATASET_METADATA_KEY = "exclusive_hitl_capture"
EXCLUSIVE_HITL_SAMPLE_METADATA_KEY = "exclusive_hitl"
EXCLUSIVE_HITL_DATASET_HASH_DOMAIN_V1 = "dexteleop.exclusive_hitl_capture.semantic.v1"
EXCLUSIVE_HITL_SAMPLE_HASH_DOMAIN_V1 = "dexteleop.exclusive_hitl_sample.semantic.v1"
EXCLUSIVE_HITL_POLICY_ARTIFACT_HASH_DOMAIN_V1 = (
    "dexteleop.exclusive_hitl_policy_artifact.semantic.v1"
)

EXCLUSIVE_HITL_POLICY_MODES = ("normal", "rtc")
EXCLUSIVE_HITL_SOURCES = ("human", "policy")
EXCLUSIVE_HITL_POLICY_SOURCE_KINDS = (
    "openpi_artifact",
    "injected_policy_emission",
    "rtc_committed_emission",
)

_POLICY_LINEAGE_KEYS = (
    "policy_emission_id",
    "policy_emission_hash",
    "policy_output_id",
    "policy_inference_id",
    "policy_target_effective_at",
    "policy_effective_until",
)
_COMPOSITION_LINEAGE_KEYS = (
    "hitl_composition_input_id",
    "hitl_composition_input_hash",
    "hitl_composition_evaluation_id",
    "hitl_composition_evaluation_hash",
    "hitl_composition_spec_hash",
    "hitl_composed_candidate_hash",
)
EXCLUSIVE_HITL_COMMAND_METADATA_KEYS = frozenset(
    {
        "control_source",
        "hitl_exclusive_source",
        "hitl_policy_mode",
        "hitl_rtc_no_action",
        "hitl_switch_id",
        "hitl_intent_id",
        "hitl_intent_hash",
        *_POLICY_LINEAGE_KEYS,
        *_COMPOSITION_LINEAGE_KEYS,
        EXCLUSIVE_HITL_SAMPLE_METADATA_KEY,
    }
)


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name=name)


def _finite_float(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _optional_positive_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _non_negative_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _checkpoint_step(value: object) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError("checkpoint_step must be non-negative.")
        return int(value)
    return _required_text(value, name="checkpoint_step")


def exclusive_hitl_policy_artifact_contract_hash(
    payload: Mapping[str, Any],
) -> str:
    """Hash the validated policy sidecar semantics, not its filesystem path."""

    return canonical_content_hash(
        dict(payload),
        domain=EXCLUSIVE_HITL_POLICY_ARTIFACT_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class ExclusiveHITLDatasetContract:
    """Dataset-wide immutable identity for one exclusive HITL configuration."""

    policy_mode: str
    initial_source: str
    active_sides: tuple[str, ...]
    policy_source_kind: str
    rtc_no_action: bool
    takeover_key: str
    policy_resume_key: str
    policy_artifact_contract_hash: str | None = None
    policy_io_layout_hash: str | None = None
    model_id: str | None = None
    checkpoint_step: int | str | None = None
    policy_prompt: str | None = None
    action_hz: float | None = None
    policy_hz: float | None = None
    action_chunk_execution_steps: int | None = None
    prediction_request_queue_threshold: int = 0
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EXCLUSIVE_HITL_DATASET_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        mode = str(self.policy_mode).strip().lower()
        if mode not in EXCLUSIVE_HITL_POLICY_MODES:
            raise ValueError("policy_mode must be 'normal' or 'rtc'.")
        source = str(self.initial_source).strip().lower()
        if source not in EXCLUSIVE_HITL_SOURCES:
            raise ValueError("initial_source must be 'human' or 'policy'.")
        sides = tuple(str(side).strip().lower() for side in self.active_sides)
        if not sides or any(side not in {"left", "right"} for side in sides):
            raise ValueError("active_sides must contain left and/or right.")
        if len(sides) != len(set(sides)):
            raise ValueError("active_sides must not contain duplicates.")
        source_kind = str(self.policy_source_kind).strip().lower()
        if source_kind not in EXCLUSIVE_HITL_POLICY_SOURCE_KINDS:
            raise ValueError("Unsupported policy_source_kind.")
        if not isinstance(self.rtc_no_action, bool):
            raise ValueError("rtc_no_action must be a boolean.")
        if mode == "normal" and self.rtc_no_action:
            raise ValueError("normal HITL cannot declare rtc_no_action.")
        if mode == "rtc" and not self.rtc_no_action:
            raise ValueError("RTC HITL v1 is explicitly no-action.")
        if mode == "rtc" and source_kind != "rtc_committed_emission":
            raise ValueError("RTC HITL must use the committed-emission source kind.")
        if mode == "normal" and source_kind == "rtc_committed_emission":
            raise ValueError("normal HITL cannot use the RTC source kind.")

        artifact_hash = _optional_text(
            self.policy_artifact_contract_hash,
            name="policy_artifact_contract_hash",
        )
        layout_hash = _optional_text(
            self.policy_io_layout_hash,
            name="policy_io_layout_hash",
        )
        if source_kind == "openpi_artifact" and (
            artifact_hash is None or layout_hash is None
        ):
            raise ValueError(
                "openpi_artifact HITL requires artifact and policy-I/O hashes."
            )
        if mode == "rtc" and (artifact_hash is not None or layout_hash is not None):
            raise ValueError(
                "RTC no-action v1 does not claim a dataset-bound policy artifact."
            )

        action_hz = _optional_positive_float(self.action_hz, name="action_hz")
        policy_hz = _optional_positive_float(self.policy_hz, name="policy_hz")
        if mode == "normal" and (action_hz is None or policy_hz is None):
            raise ValueError("normal HITL requires action_hz and policy_hz.")
        if mode == "rtc" and (action_hz is not None or policy_hz is not None):
            raise ValueError("RTC no-action v1 does not own normal policy rates.")
        execution_steps = _optional_positive_int(
            self.action_chunk_execution_steps,
            name="action_chunk_execution_steps",
        )
        threshold = _non_negative_int(
            self.prediction_request_queue_threshold,
            name="prediction_request_queue_threshold",
        )
        if mode == "rtc" and (execution_steps is not None or threshold != 0):
            raise ValueError("RTC no-action v1 does not own normal chunk scheduling.")

        object.__setattr__(self, "policy_mode", mode)
        object.__setattr__(self, "initial_source", source)
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "policy_source_kind", source_kind)
        takeover_key = _required_text(self.takeover_key, name="takeover_key")
        resume_key = _required_text(
            self.policy_resume_key,
            name="policy_resume_key",
        )
        if takeover_key == resume_key:
            raise ValueError("takeover_key and policy_resume_key must differ.")
        object.__setattr__(self, "takeover_key", takeover_key)
        object.__setattr__(
            self,
            "policy_resume_key",
            resume_key,
        )
        object.__setattr__(self, "policy_artifact_contract_hash", artifact_hash)
        object.__setattr__(self, "policy_io_layout_hash", layout_hash)
        object.__setattr__(
            self, "model_id", _optional_text(self.model_id, name="model_id")
        )
        object.__setattr__(
            self, "checkpoint_step", _checkpoint_step(self.checkpoint_step)
        )
        object.__setattr__(
            self,
            "policy_prompt",
            _optional_text(self.policy_prompt, name="policy_prompt"),
        )
        object.__setattr__(self, "action_hz", action_hz)
        object.__setattr__(self, "policy_hz", policy_hz)
        object.__setattr__(self, "action_chunk_execution_steps", execution_steps)
        object.__setattr__(self, "prediction_request_queue_threshold", threshold)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=EXCLUSIVE_HITL_DATASET_HASH_DOMAIN_V1,
            ),
        )

    @property
    def policy_identity_status(self) -> str:
        if self.policy_mode == "rtc":
            return "rtc_no_action"
        if self.policy_artifact_contract_hash is None:
            return "unbound_injected_source"
        return "artifact_bound"

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "selection_semantics": "exclusive_single_source_no_blending",
            "recorded_action_semantics": "selected_executed_action_only",
            "counterfactual_policy_action_recorded": False,
            "policy_mode": self.policy_mode,
            "initial_source": self.initial_source,
            "active_sides": list(self.active_sides),
            "policy_source_kind": self.policy_source_kind,
            "policy_identity_status": self.policy_identity_status,
            "rtc_no_action": self.rtc_no_action,
            "takeover_key": self.takeover_key,
            "policy_resume_key": self.policy_resume_key,
            "sample_metadata_key": EXCLUSIVE_HITL_SAMPLE_METADATA_KEY,
            "policy_artifact_contract_hash": self.policy_artifact_contract_hash,
            "policy_io_layout_hash": self.policy_io_layout_hash,
            "model_id": self.model_id,
            "checkpoint_step": self.checkpoint_step,
            "policy_prompt": self.policy_prompt,
            "action_hz": self.action_hz,
            "policy_hz": self.policy_hz,
            "action_chunk_execution_steps": self.action_chunk_execution_steps,
            "prediction_request_queue_threshold": self.prediction_request_queue_threshold,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExclusiveHITLDatasetContract":
        if payload.get("schema") != EXCLUSIVE_HITL_DATASET_SCHEMA_V1:
            raise ValueError("Unsupported exclusive HITL dataset schema.")
        if payload.get("selection_semantics") != "exclusive_single_source_no_blending":
            raise ValueError("Exclusive HITL selection semantics mismatch.")
        if payload.get("recorded_action_semantics") != "selected_executed_action_only":
            raise ValueError("Exclusive HITL recorded-action semantics mismatch.")
        if payload.get("counterfactual_policy_action_recorded") is not False:
            raise ValueError(
                "Exclusive HITL v1 does not record counterfactual actions."
            )
        if payload.get("sample_metadata_key") != EXCLUSIVE_HITL_SAMPLE_METADATA_KEY:
            raise ValueError("Exclusive HITL sample metadata key mismatch.")
        active_sides = payload.get("active_sides")
        if not isinstance(active_sides, (list, tuple)):
            raise ValueError("exclusive HITL active_sides must be a sequence.")
        result = cls(
            policy_mode=payload.get("policy_mode", ""),
            initial_source=payload.get("initial_source", ""),
            active_sides=tuple(active_sides),
            policy_source_kind=payload.get("policy_source_kind", ""),
            rtc_no_action=payload.get("rtc_no_action"),
            takeover_key=payload.get("takeover_key", ""),
            policy_resume_key=payload.get("policy_resume_key", ""),
            policy_artifact_contract_hash=payload.get("policy_artifact_contract_hash"),
            policy_io_layout_hash=payload.get("policy_io_layout_hash"),
            model_id=payload.get("model_id"),
            checkpoint_step=payload.get("checkpoint_step"),
            policy_prompt=payload.get("policy_prompt"),
            action_hz=payload.get("action_hz"),
            policy_hz=payload.get("policy_hz"),
            action_chunk_execution_steps=payload.get("action_chunk_execution_steps"),
            prediction_request_queue_threshold=payload.get(
                "prediction_request_queue_threshold", 0
            ),
        )
        declared_status = payload.get("policy_identity_status")
        if declared_status != result.policy_identity_status:
            raise ValueError("Exclusive HITL policy identity status mismatch.")
        declared_hash = payload.get("semantic_content_hash")
        if (
            declared_hash is not None
            and str(declared_hash) != result.semantic_content_hash
        ):
            raise ValueError("Exclusive HITL dataset semantic hash mismatch.")
        return result


@dataclass(frozen=True)
class ExclusiveHITLSampleLineage:
    """Causal selected-source lineage for one persisted action sample."""

    source: str
    policy_mode: str
    rtc_no_action: bool
    switch_id: str | None = None
    intent_id: str | None = None
    intent_hash: str | None = None
    policy_emission_id: str | None = None
    policy_emission_hash: str | None = None
    policy_output_id: str | None = None
    policy_inference_id: str | None = None
    policy_target_effective_at: float | None = None
    policy_effective_until: float | None = None
    composition_input_id: str | None = None
    composition_input_hash: str | None = None
    composition_evaluation_id: str | None = None
    composition_evaluation_hash: str | None = None
    composition_spec_hash: str | None = None
    composed_candidate_hash: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EXCLUSIVE_HITL_SAMPLE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        source = str(self.source).strip().lower()
        mode = str(self.policy_mode).strip().lower()
        if source not in EXCLUSIVE_HITL_SOURCES:
            raise ValueError("exclusive HITL sample source must be human or policy.")
        if mode not in EXCLUSIVE_HITL_POLICY_MODES:
            raise ValueError("exclusive HITL sample policy_mode must be normal or rtc.")
        if not isinstance(self.rtc_no_action, bool):
            raise ValueError("exclusive HITL sample rtc_no_action must be a boolean.")
        if mode == "normal" and self.rtc_no_action:
            raise ValueError("normal HITL sample cannot declare rtc_no_action.")
        if mode == "rtc" and not self.rtc_no_action:
            raise ValueError("RTC HITL v1 sample must remain no-action.")
        if source == "policy" and mode == "rtc":
            raise ValueError("RTC HITL v1 cannot persist a policy-owned action.")

        policy_values = {
            name: getattr(self, name)
            for name in (
                "policy_emission_id",
                "policy_emission_hash",
                "policy_output_id",
                "policy_inference_id",
                "composition_input_id",
                "composition_input_hash",
                "composition_evaluation_id",
                "composition_evaluation_hash",
                "composition_spec_hash",
                "composed_candidate_hash",
            )
        }
        if source == "policy":
            normalized_policy = {
                name: _required_text(value, name=name)
                for name, value in policy_values.items()
            }
            target_at = _finite_float(
                self.policy_target_effective_at,
                name="policy_target_effective_at",
            )
            effective_until = _finite_float(
                self.policy_effective_until,
                name="policy_effective_until",
            )
            if effective_until < target_at:
                raise ValueError(
                    "policy_effective_until must not precede policy_target_effective_at."
                )
        else:
            if any(value is not None for value in policy_values.values()) or (
                self.policy_target_effective_at is not None
                or self.policy_effective_until is not None
            ):
                raise ValueError(
                    "human-owned HITL samples cannot carry policy action lineage."
                )
            normalized_policy = {name: None for name in policy_values}
            target_at = None
            effective_until = None

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "policy_mode", mode)
        for name in ("switch_id", "intent_id", "intent_hash"):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), name=name),
            )
        for name, value in normalized_policy.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "policy_target_effective_at", target_at)
        object.__setattr__(self, "policy_effective_until", effective_until)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=EXCLUSIVE_HITL_SAMPLE_HASH_DOMAIN_V1,
            ),
        )

    @property
    def policy_lineage_complete(self) -> bool:
        return self.source != "policy" or self.policy_emission_id is not None

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        switch = None
        if any(
            value is not None
            for value in (self.switch_id, self.intent_id, self.intent_hash)
        ):
            switch = {
                "switch_id": self.switch_id,
                "intent_id": self.intent_id,
                "intent_hash": self.intent_hash,
            }
        policy_emission = None
        composition = None
        if self.source == "policy":
            policy_emission = {
                "emission_id": self.policy_emission_id,
                "emission_hash": self.policy_emission_hash,
                "output_id": self.policy_output_id,
                "inference_id": self.policy_inference_id,
                "target_effective_at": self.policy_target_effective_at,
                "effective_until": self.policy_effective_until,
            }
            composition = {
                "input_id": self.composition_input_id,
                "input_hash": self.composition_input_hash,
                "evaluation_id": self.composition_evaluation_id,
                "evaluation_hash": self.composition_evaluation_hash,
                "spec_hash": self.composition_spec_hash,
                "composed_candidate_hash": self.composed_candidate_hash,
            }
        payload: dict[str, Any] = {
            "schema": self.schema,
            "source": self.source,
            "policy_mode": self.policy_mode,
            "rtc_no_action": self.rtc_no_action,
            "switch": switch,
            "policy_emission": policy_emission,
            "composition": composition,
            "policy_lineage_complete": self.policy_lineage_complete,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    def to_command_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "control_source": self.source,
            "hitl_exclusive_source": self.source,
            "hitl_policy_mode": self.policy_mode,
            "hitl_rtc_no_action": self.rtc_no_action,
            EXCLUSIVE_HITL_SAMPLE_METADATA_KEY: self.to_payload(),
        }
        optional = {
            "hitl_switch_id": self.switch_id,
            "hitl_intent_id": self.intent_id,
            "hitl_intent_hash": self.intent_hash,
            "policy_emission_id": self.policy_emission_id,
            "policy_emission_hash": self.policy_emission_hash,
            "policy_output_id": self.policy_output_id,
            "policy_inference_id": self.policy_inference_id,
            "policy_target_effective_at": self.policy_target_effective_at,
            "policy_effective_until": self.policy_effective_until,
            "hitl_composition_input_id": self.composition_input_id,
            "hitl_composition_input_hash": self.composition_input_hash,
            "hitl_composition_evaluation_id": self.composition_evaluation_id,
            "hitl_composition_evaluation_hash": self.composition_evaluation_hash,
            "hitl_composition_spec_hash": self.composition_spec_hash,
            "hitl_composed_candidate_hash": self.composed_candidate_hash,
        }
        metadata.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return metadata

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExclusiveHITLSampleLineage":
        if payload.get("schema") != EXCLUSIVE_HITL_SAMPLE_SCHEMA_V1:
            raise ValueError("Unsupported exclusive HITL sample schema.")
        switch = payload.get("switch")
        if switch is not None and not isinstance(switch, Mapping):
            raise ValueError("exclusive HITL sample switch must be an object or null.")
        emission = payload.get("policy_emission")
        composition = payload.get("composition")
        if emission is not None and not isinstance(emission, Mapping):
            raise ValueError(
                "exclusive HITL policy_emission must be an object or null."
            )
        if composition is not None and not isinstance(composition, Mapping):
            raise ValueError("exclusive HITL composition must be an object or null.")
        switch = {} if switch is None else switch
        emission = {} if emission is None else emission
        composition = {} if composition is None else composition
        result = cls(
            source=payload.get("source", ""),
            policy_mode=payload.get("policy_mode", ""),
            rtc_no_action=payload.get("rtc_no_action"),
            switch_id=switch.get("switch_id"),
            intent_id=switch.get("intent_id"),
            intent_hash=switch.get("intent_hash"),
            policy_emission_id=emission.get("emission_id"),
            policy_emission_hash=emission.get("emission_hash"),
            policy_output_id=emission.get("output_id"),
            policy_inference_id=emission.get("inference_id"),
            policy_target_effective_at=emission.get("target_effective_at"),
            policy_effective_until=emission.get("effective_until"),
            composition_input_id=composition.get("input_id"),
            composition_input_hash=composition.get("input_hash"),
            composition_evaluation_id=composition.get("evaluation_id"),
            composition_evaluation_hash=composition.get("evaluation_hash"),
            composition_spec_hash=composition.get("spec_hash"),
            composed_candidate_hash=composition.get("composed_candidate_hash"),
        )
        if payload.get("policy_lineage_complete") is not result.policy_lineage_complete:
            raise ValueError("exclusive HITL policy lineage completeness mismatch.")
        declared_hash = payload.get("semantic_content_hash")
        if (
            declared_hash is not None
            and str(declared_hash) != result.semantic_content_hash
        ):
            raise ValueError("Exclusive HITL sample semantic hash mismatch.")
        return result

    @classmethod
    def from_command_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
    ) -> "ExclusiveHITLSampleLineage | None":
        if not metadata:
            return None
        embedded = metadata.get(EXCLUSIVE_HITL_SAMPLE_METADATA_KEY)
        embedded_record = None
        if embedded is not None:
            if not isinstance(embedded, Mapping):
                raise ValueError(
                    "exclusive HITL embedded sample metadata must be an object."
                )
            embedded_record = cls.from_payload(embedded)

        source_value = metadata.get("hitl_exclusive_source")
        if source_value is None:
            related = any(
                key in metadata
                for key in EXCLUSIVE_HITL_COMMAND_METADATA_KEYS
                if key != EXCLUSIVE_HITL_SAMPLE_METADATA_KEY
            )
            if related:
                raise ValueError(
                    "partial HITL command metadata is missing hitl_exclusive_source."
                )
            return embedded_record
        control_source = metadata.get("control_source")
        if (
            control_source is None
            or str(control_source).strip().lower() != str(source_value).strip().lower()
        ):
            raise ValueError("control_source must match hitl_exclusive_source.")
        record = cls(
            source=source_value,
            policy_mode=metadata.get("hitl_policy_mode", ""),
            rtc_no_action=metadata.get("hitl_rtc_no_action"),
            switch_id=metadata.get("hitl_switch_id"),
            intent_id=metadata.get("hitl_intent_id"),
            intent_hash=metadata.get("hitl_intent_hash"),
            policy_emission_id=metadata.get("policy_emission_id"),
            policy_emission_hash=metadata.get("policy_emission_hash"),
            policy_output_id=metadata.get("policy_output_id"),
            policy_inference_id=metadata.get("policy_inference_id"),
            policy_target_effective_at=metadata.get("policy_target_effective_at"),
            policy_effective_until=metadata.get("policy_effective_until"),
            composition_input_id=metadata.get("hitl_composition_input_id"),
            composition_input_hash=metadata.get("hitl_composition_input_hash"),
            composition_evaluation_id=metadata.get("hitl_composition_evaluation_id"),
            composition_evaluation_hash=metadata.get(
                "hitl_composition_evaluation_hash"
            ),
            composition_spec_hash=metadata.get("hitl_composition_spec_hash"),
            composed_candidate_hash=metadata.get("hitl_composed_candidate_hash"),
        )
        if embedded_record is not None and embedded_record != record:
            raise ValueError(
                "embedded exclusive HITL record disagrees with command metadata."
            )
        return record


__all__ = [
    "EXCLUSIVE_HITL_COMMAND_METADATA_KEYS",
    "EXCLUSIVE_HITL_DATASET_HASH_DOMAIN_V1",
    "EXCLUSIVE_HITL_DATASET_METADATA_KEY",
    "EXCLUSIVE_HITL_DATASET_SCHEMA_V1",
    "EXCLUSIVE_HITL_POLICY_ARTIFACT_HASH_DOMAIN_V1",
    "EXCLUSIVE_HITL_POLICY_MODES",
    "EXCLUSIVE_HITL_POLICY_SOURCE_KINDS",
    "EXCLUSIVE_HITL_SAMPLE_HASH_DOMAIN_V1",
    "EXCLUSIVE_HITL_SAMPLE_METADATA_KEY",
    "EXCLUSIVE_HITL_SAMPLE_SCHEMA_V1",
    "EXCLUSIVE_HITL_SOURCES",
    "ExclusiveHITLDatasetContract",
    "ExclusiveHITLSampleLineage",
    "exclusive_hitl_policy_artifact_contract_hash",
]
