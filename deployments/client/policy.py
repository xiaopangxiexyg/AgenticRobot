"""Hardware-neutral policy loading and prediction adapter."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from robot_motion.deploy.observation import DeploymentObservation


@dataclass
class LoadedPolicy:
    """Small adapter around user policies with a stable ``predict()`` method."""

    policy: Any
    name: str
    call_with_model_input: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def predict(self, observation: DeploymentObservation) -> Any:
        predict = getattr(self.policy, "predict", None)
        if callable(predict):
            return predict(observation)
        if callable(self.policy):
            if self.call_with_model_input:
                return self.policy(observation.model_input())
            return self.policy(observation)
        raise TypeError(f"Loaded policy {self.name!r} is not callable and has no predict().")

    def begin_prediction_transaction(self) -> Any:
        """Capture optional mutable policy state before one prediction."""

        begin = getattr(self.policy, "begin_prediction_transaction", None)
        return begin() if callable(begin) else None

    def commit_prediction_transaction(self, checkpoint: Any) -> None:
        commit = getattr(self.policy, "commit_prediction_transaction", None)
        if callable(commit):
            commit(checkpoint)

    def rollback_prediction_transaction(self, checkpoint: Any) -> None:
        rollback = getattr(self.policy, "rollback_prediction_transaction", None)
        if callable(rollback):
            rollback(checkpoint)


class HoldPolicy:
    """Built-in policy that returns the latest observed robot state."""

    def predict(self, observation: DeploymentObservation) -> dict[str, Any]:
        return {
            "action": observation.hold_action(),
            "metadata": {"policy": "hold"},
        }


def _load_module_from_path(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"_dexteleop_deployment_policy_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import policy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module(module_ref: str) -> ModuleType:
    path = Path(module_ref)
    if module_ref.endswith(".py") or path.exists():
        return _load_module_from_path(path.expanduser().resolve())
    return importlib.import_module(module_ref)


def _split_spec(spec: str) -> tuple[str, str | None]:
    text = str(spec).strip()
    if ":" not in text:
        return text, None
    module_ref, object_name = text.rsplit(":", 1)
    return module_ref.strip(), object_name.strip() or None


def _default_policy_object(module: ModuleType) -> Any:
    for name in ("create_policy", "Policy", "policy", "predict"):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(
        f"Policy module {module.__name__!r} must expose create_policy, Policy, policy, or predict."
    )


def _construct_policy(obj: Any, kwargs: Mapping[str, Any]) -> tuple[Any, bool]:
    call_with_model_input = False
    if isinstance(obj, type):
        return obj(**dict(kwargs)), call_with_model_input
    if callable(obj) and getattr(obj, "__name__", "") == "create_policy":
        return obj(**dict(kwargs)), call_with_model_input
    if callable(obj) and getattr(obj, "__name__", "") == "predict":
        call_with_model_input = True
        return obj, call_with_model_input
    return obj, call_with_model_input


def load_policy(
    spec: str | None,
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> LoadedPolicy:
    """Load a policy from a built-in name, module, or Python file."""

    text = str(spec or "hold").strip()
    payload = {} if kwargs is None else dict(kwargs)
    if not text or text.lower() in {"hold", "builtin:hold"}:
        return LoadedPolicy(policy=HoldPolicy(), name="hold", metadata={"builtin": True})

    module_ref, object_name = _split_spec(text)
    module = _load_module(module_ref)
    obj = getattr(module, object_name) if object_name else _default_policy_object(module)
    policy, call_with_model_input = _construct_policy(obj, payload)
    name = f"{module_ref}:{object_name}" if object_name else module_ref
    return LoadedPolicy(
        policy=policy,
        name=name,
        call_with_model_input=call_with_model_input,
        metadata={"module": module_ref, "object": object_name, "kwargs": payload},
    )


__all__ = ["HoldPolicy", "LoadedPolicy", "load_policy"]
