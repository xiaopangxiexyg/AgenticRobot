from __future__ import annotations

from collections.abc import Mapping

from ..interfaces import EffectorBackend, Rig
from ..schemas import TeleopCommand


class BimanualRig(Rig):
    def __init__(self, backends: Mapping[str, EffectorBackend], *, name: str = "bimanual_rig"):
        self.backends = dict(backends)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        for backend in self.backends.values():
            backend.start()

    def is_ready(self) -> bool:
        return all(backend.is_ready() for backend in self.backends.values())

    def send_commands(self, command: TeleopCommand) -> None:
        for effector_name, effector_command in command.commands.items():
            if effector_name not in self.backends:
                raise KeyError(f"Rig backend missing for effector '{effector_name}'.")
            self.backends[effector_name].send_command(effector_command)

    def close(self) -> None:
        for backend in self.backends.values():
            backend.close()
