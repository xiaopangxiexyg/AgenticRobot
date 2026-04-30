from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from project_paths import teleop_records_root


@dataclass
class RecordingConfig:
    enabled: bool = True
    save_raw_inputs: bool = True
    save_calibration: bool = True
    save_commands: bool = True
    save_fault_events: bool = True


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_serializable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


class SessionRecorder:
    def __init__(self, root_dir: str | Path | None = None, config: RecordingConfig | None = None):
        self.config = config if config is not None else RecordingConfig()
        if root_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            root_dir = teleop_records_root() / stamp
        self.root_dir = Path(root_dir)
        self._handles: dict[str, TextIO] = {}
        if self.config.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def _open(self, name: str) -> TextIO:
        if name not in self._handles:
            path = self.root_dir / f"{name}.jsonl"
            self._handles[name] = path.open("a", encoding="utf-8")
        return self._handles[name]

    def _write_jsonl(self, stream_name: str, payload: Any) -> None:
        if not self.config.enabled:
            return
        handle = self._open(stream_name)
        handle.write(json.dumps(_to_serializable(payload), ensure_ascii=True) + "\n")
        handle.flush()

    def record_channels(self, channels: dict[str, Any], timestamp: float | None = None) -> None:
        if not (self.config.enabled and self.config.save_raw_inputs):
            return
        self._write_jsonl(
            "raw_inputs",
            {"timestamp": time.time() if timestamp is None else timestamp, "channels": channels},
        )

    def record_command(self, command: Any) -> None:
        if not (self.config.enabled and self.config.save_commands):
            return
        self._write_jsonl("commands", command)

    def record_fault(self, fault: Any) -> None:
        if not (self.config.enabled and self.config.save_fault_events):
            return
        self._write_jsonl("faults", fault)

    def record_calibration(self, calibration: Any) -> None:
        if not (self.config.enabled and self.config.save_calibration):
            return
        path = self.root_dir / "calibration.json"
        path.write_text(json.dumps(_to_serializable(calibration), indent=2), encoding="utf-8")

    def record_state_transition(self, old_state: str, new_state: str, reason: str = "") -> None:
        if not self.config.enabled:
            return
        self._write_jsonl(
            "state_transitions",
            {
                "timestamp": time.time(),
                "old_state": old_state,
                "new_state": new_state,
                "reason": reason,
            },
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
