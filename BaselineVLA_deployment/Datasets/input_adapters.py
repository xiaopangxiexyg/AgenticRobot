from __future__ import annotations

from typing import Any, Dict

import torch

from .base import PreparedBatch


def _move_mapping_to_device(mapping: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in mapping.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = _move_mapping_to_device(value, device)
        else:
            moved[key] = value
    return moved


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return _move_mapping_to_device(value, device)
    return value


class BaseInputAdapter:
    name = "base"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model_family = str(getattr(config.model, "family", "mlp")).lower()
        task_config = getattr(config.model, "task", None)
        self.use_task_input = bool(getattr(task_config, "enabled", False)) if task_config is not None else False

    def _prepare_task_inputs(self, observations: Dict[str, Any], metadata: Any, device: torch.device) -> Dict[str, Any]:
        if not self.use_task_input:
            return {}

        if self.model_family == "vla":
            task_texts = observations.get("task_text")
            if task_texts is None:
                task_texts = [str(sample_metadata.get("task_text", "unknown_task")) for sample_metadata in metadata]
            return {"task_text": list(task_texts)}

        task_ids = observations.get("task_id")
        if task_ids is None:
            task_ids = torch.zeros(len(metadata), dtype=torch.long, device=device)
        return {"task_id": task_ids.to(device=device, dtype=torch.long)}

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        raise NotImplementedError


class VanillaInputAdapter(BaseInputAdapter):
    name = "vanilla"

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        observations = _move_mapping_to_device(batch["observations"], device)
        targets = _move_mapping_to_device(batch["targets"], device)
        availability = _move_mapping_to_device(batch["availability"], device)
        model_inputs = dict(observations)
        model_inputs.update(self._prepare_task_inputs(observations, batch["metadata"], device))
        return PreparedBatch(
            model_inputs=model_inputs,
            targets={
                "action": targets["action"],
            },
            metadata=batch["metadata"],
            availability=availability,
        )


class VanillaTactileInputAdapter(BaseInputAdapter):
    name = "vanilla_tactile"

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        observations = _move_mapping_to_device(batch["observations"], device)
        targets = _move_mapping_to_device(batch["targets"], device)
        availability = _move_mapping_to_device(batch["availability"], device)
        if "tactile" not in observations:
            raise KeyError(
                "VanillaTactileInputAdapter requires observations['tactile']. "
                "Enable raw.sequence.tactile in the data config."
            )
        model_inputs = dict(observations)
        model_inputs.update(self._prepare_task_inputs(observations, batch["metadata"], device))
        return PreparedBatch(
            model_inputs=model_inputs,
            targets={
                "action": targets["action"],
            },
            metadata=batch["metadata"],
            availability=availability,
        )


class GTMedInputAdapter(BaseInputAdapter):
    name = "gtmed"

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        observations = _move_mapping_to_device(batch["observations"], device)
        targets = _move_mapping_to_device(batch["targets"], device)
        availability = _move_mapping_to_device(batch["availability"], device)
        med_value = batch["intermediates"].get("med")
        if med_value is None:
            raise KeyError("GTMedInputAdapter requires batch['intermediates']['med'].")
        model_inputs = dict(observations)
        model_inputs.update(self._prepare_task_inputs(observations, batch["metadata"], device))
        model_inputs["med"] = _move_value_to_device(med_value, device)
        return PreparedBatch(
            model_inputs=model_inputs,
            targets={
                "action": targets["action"],
            },
            metadata=batch["metadata"],
            availability=availability,
        )


class FullTrainInputAdapter(BaseInputAdapter):
    name = "fulltrain"

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        observations = _move_mapping_to_device(batch["observations"], device)
        targets = _move_mapping_to_device(batch["targets"], device)
        availability = _move_mapping_to_device(batch["availability"], device)
        med_value = batch["intermediates"].get("med")
        if med_value is None:
            raise KeyError("FullTrainInputAdapter requires batch['intermediates']['med'].")
        model_inputs = dict(observations)
        model_inputs.update(self._prepare_task_inputs(observations, batch["metadata"], device))
        return PreparedBatch(
            model_inputs=model_inputs,
            targets={
                "action": targets["action"],
                "med": _move_value_to_device(med_value, device),
            },
            metadata=batch["metadata"],
            availability=availability,
        )


class GenMedInputAdapter(BaseInputAdapter):
    name = "genmed"

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        observations = _move_mapping_to_device(batch["observations"], device)
        availability = _move_mapping_to_device(batch["availability"], device)
        med_value = batch["intermediates"].get("med")
        if med_value is None:
            raise KeyError("GenMedInputAdapter requires batch['intermediates']['med'].")
        model_inputs = dict(observations)
        model_inputs.update(self._prepare_task_inputs(observations, batch["metadata"], device))
        return PreparedBatch(
            model_inputs=model_inputs,
            targets={
                "med": _move_value_to_device(med_value, device),
            },
            metadata=batch["metadata"],
            availability=availability,
        )


INPUT_ADAPTERS = {
    "vanilla": VanillaInputAdapter,
    "vanilla_tactile": VanillaTactileInputAdapter,
    "gtmed": GTMedInputAdapter,
    "fulltrain": FullTrainInputAdapter,
    "genmed": GenMedInputAdapter,
}
