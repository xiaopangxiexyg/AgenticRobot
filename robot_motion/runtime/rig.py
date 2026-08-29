from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .contracts import TeleopCommand
from .dispatch import RigDispatchError
from .interfaces import EffectorBackend, Rig
from .lifecycle import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    run_best_effort_cleanup,
)


class BimanualRig(Rig):
    def __init__(
        self,
        backends: Mapping[str, EffectorBackend],
        *,
        name: str = "bimanual_rig",
        parallel_dispatch: bool = False,
        parallel_workers: int | None = None,
        fail_stop_on_dispatch_error: bool = False,
    ):
        self.backends = dict(backends)
        self._name = name
        self.fail_stop_on_dispatch_error = bool(fail_stop_on_dispatch_error)
        self._started = False
        self._terminal = False
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
        if self._started:
            return
        if self._terminal:
            raise RuntimeError(
                f"{self.name} was closed or rolled back and cannot be started "
                "again; construct a fresh rig."
            )
        attempted: list[tuple[str, EffectorBackend]] = []
        try:
            for effector_name, backend in self.backends.items():
                attempted.append((effector_name, backend))
                backend.start()
        except BaseException as start_error:  # noqa: BLE001 - rollback before propagating.
            self._terminal = True
            cleanup_failures = collect_lifecycle_failures(
                (
                    (f"backend[{effector_name}]", backend.close)
                    for effector_name, backend in reversed(attempted)
                )
            )
            if cleanup_failures:
                raise LifecycleStartupError(
                    f"{self.name}.start",
                    start_error,
                    cleanup_failures,
                ) from start_error
            raise
        self._started = True

    def is_ready(self) -> bool:
        return all(backend.is_ready() for backend in self.backends.values())

    def send_commands(
        self,
        command: TeleopCommand,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> dict[str, object | None]:
        missing_effectors = [
            effector_name
            for effector_name in command.commands
            if effector_name not in self.backends
        ]
        if missing_effectors:
            raise KeyError(f"Rig backend missing for effectors {missing_effectors}.")

        # Preserve one atomic parent timestamp across parallel backend
        # dispatch. Consumers that quantize a bimanual command into a slower
        # source cut must never let two worker start times land on opposite
        # sides of a bucket boundary.
        parent_command_timestamp = (
            time.time()
            if getattr(command, "timestamp", None) is None
            else float(command.timestamp)
        )
        for effector_command in command.commands.values():
            metadata = getattr(effector_command, "metadata", None)
            if isinstance(metadata, dict):
                metadata.setdefault(
                    "_parent_command_timestamp",
                    parent_command_timestamp,
                )

        def send_one(effector_name: str, effector_command) -> tuple[float, object | None]:
            started_at = time.perf_counter()
            receipt = self.backends[effector_name].send_command(effector_command)
            return time.perf_counter() - started_at, receipt

        receipts: dict[str, object | None] = {}
        if self._executor is not None and len(command.commands) > 1:
            futures = []
            for effector_name, effector_command in command.commands.items():
                futures.append(
                    (
                        effector_name,
                        self._executor.submit(send_one, effector_name, effector_command),
                    )
                )
            failures: list[tuple[str, Exception]] = []
            for effector_name, future in futures:
                try:
                    elapsed_sec, receipt = future.result()
                except Exception as exc:  # noqa: BLE001 - group fail-stop must see all failures.
                    failures.append((effector_name, exc))
                    continue
                receipts[effector_name] = receipt
                if phase_recorder is not None:
                    phase_recorder(f"rig.{effector_name}.send_command", elapsed_sec)
            if failures:
                self._raise_dispatch_failures(failures)
            return receipts
        for effector_name, effector_command in command.commands.items():
            try:
                elapsed_sec, receipt = send_one(effector_name, effector_command)
            except Exception as exc:  # noqa: BLE001 - preserve optional group fail-stop.
                self._raise_dispatch_failures([(effector_name, exc)])
                raise AssertionError("unreachable")
            receipts[effector_name] = receipt
            if phase_recorder is not None:
                phase_recorder(
                    f"rig.{effector_name}.send_command",
                    elapsed_sec,
                )
        return receipts

    def _raise_dispatch_failures(
        self,
        failures: list[tuple[str, Exception]],
    ) -> None:
        if not self.fail_stop_on_dispatch_error:
            raise failures[0][1]
        failure_text = "; ".join(
            f"{name}: {type(exc).__name__}: {exc}" for name, exc in failures
        )
        hold_error: Exception | None = None
        try:
            self.hold_motion(f"dispatch failure: {failure_text}")
        except Exception as exc:  # noqa: BLE001 - report dispatch and fail-stop failures together.
            hold_error = exc
        raise RigDispatchError(
            rig_name=self.name,
            failures=dict(failures),
            hold_error=hold_error,
        ) from failures[0][1]

    def hold_motion(self, reason: str = "manual hold") -> None:
        failures: list[tuple[str, Exception]] = []
        executor = self._executor
        if executor is not None and len(self.backends) > 1:
            futures = [
                (effector_name, executor.submit(backend.hold_motion, reason))
                for effector_name, backend in self.backends.items()
            ]
            for effector_name, future in futures:
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((effector_name, exc))
        else:
            for effector_name, backend in self.backends.items():
                try:
                    backend.hold_motion(reason)
                except Exception as exc:  # noqa: BLE001
                    failures.append((effector_name, exc))
        if failures:
            details = "; ".join(
                f"{effector_name}: {type(exc).__name__}: {exc}"
                for effector_name, exc in failures
            )
            raise RuntimeError(
                f"{self.name} hold_motion failed for {len(failures)} backend(s): {details}"
            )

    def close(self) -> None:
        self._terminal = True
        self._started = False
        executor = self._executor
        self._executor = None
        steps: list[tuple[str, Callable[[], None]]] = []
        if executor is not None:
            steps.append(("dispatch_executor", lambda: executor.shutdown(wait=True)))
        steps.extend(
            (f"backend[{effector_name}]", backend.close)
            for effector_name, backend in reversed(tuple(self.backends.items()))
        )
        run_best_effort_cleanup(f"{self.name}.close", steps)


CompositeRig = BimanualRig


__all__ = ["BimanualRig", "CompositeRig"]
