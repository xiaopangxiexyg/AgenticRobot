from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

from ..interfaces import EffectorBackend, Rig
from ..schemas import TeleopCommand


class BimanualRig(Rig):
    def __init__(
        self,
        backends: Mapping[str, EffectorBackend],
        *,
        name: str = "bimanual_rig",
        parallel_dispatch: bool = False,
        parallel_workers: int | None = None,
    ):
        self.backends = dict(backends)
        self._name = name
        worker_count = len(self.backends) if parallel_workers is None else max(1, int(parallel_workers))
        self._executor = (
            ThreadPoolExecutor(
                max_workers=min(worker_count, max(1, len(self.backends))),
                thread_name_prefix=f"{name}_dispatch",
            )
            if parallel_dispatch and len(self.backends) > 1
            else None
        )

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        for backend in self.backends.values():
            backend.start()

    def is_ready(self) -> bool:
        return all(backend.is_ready() for backend in self.backends.values())

    def send_commands(self, command: TeleopCommand) -> None:
        if self._executor is not None and len(command.commands) > 1:
            futures = []
            for effector_name, effector_command in command.commands.items():
                if effector_name not in self.backends:
                    raise KeyError(f"Rig backend missing for effector '{effector_name}'.")
                futures.append(
                    self._executor.submit(
                        self.backends[effector_name].send_command,
                        effector_command,
                    )
                )
            for future in futures:
                future.result()
            return
        for effector_name, effector_command in command.commands.items():
            if effector_name not in self.backends:
                raise KeyError(f"Rig backend missing for effector '{effector_name}'.")
            self.backends[effector_name].send_command(effector_command)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for backend in self.backends.values():
            backend.close()
