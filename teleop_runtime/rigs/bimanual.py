from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

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

    def send_commands(
        self,
        command: TeleopCommand,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> None:
        def send_one(effector_name: str, effector_command) -> float:
            started_at = time.perf_counter()
            self.backends[effector_name].send_command(effector_command)
            return time.perf_counter() - started_at

        if self._executor is not None and len(command.commands) > 1:
            futures = []
            for effector_name, effector_command in command.commands.items():
                if effector_name not in self.backends:
                    raise KeyError(f"Rig backend missing for effector '{effector_name}'.")
                futures.append(
                    (
                        effector_name,
                        self._executor.submit(send_one, effector_name, effector_command),
                    )
                )
            for effector_name, future in futures:
                elapsed_sec = future.result()
                if phase_recorder is not None:
                    phase_recorder(
                        f"rig.{effector_name}.send_command",
                        elapsed_sec,
                    )
            return
        for effector_name, effector_command in command.commands.items():
            if effector_name not in self.backends:
                raise KeyError(f"Rig backend missing for effector '{effector_name}'.")
            elapsed_sec = send_one(effector_name, effector_command)
            if phase_recorder is not None:
                phase_recorder(
                    f"rig.{effector_name}.send_command",
                    elapsed_sec,
                )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for backend in self.backends.values():
            backend.close()
