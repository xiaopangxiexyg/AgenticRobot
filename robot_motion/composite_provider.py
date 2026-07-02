from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from .interfaces import KinematicsProvider
from .types import IKCandidate, SingularityReport


@dataclass(frozen=True)
class _ProviderResult:
    index: int
    name: str
    elapsed_ms: float
    candidates: list[IKCandidate]
    error: str | None = None


class CompositeKinematicsProvider:
    """Merge multiple robot-compatible kinematics providers.

    Candidate selection stays in :class:`IKSafetyResolver`; this adapter only
    asks every provider for candidates, annotates diagnostics, and deduplicates
    identical joint vectors.
    """

    def __init__(
        self,
        providers: tuple[KinematicsProvider, ...] | list[KinematicsProvider],
        *,
        names: tuple[str, ...] | list[str] | None = None,
        primary_index: int = 0,
    ):
        if not providers:
            raise ValueError("CompositeKinematicsProvider requires at least one provider")
        self.providers = tuple(providers)
        if names is None:
            self.names = tuple(type(provider).__name__ for provider in self.providers)
        else:
            self.names = tuple(str(name) for name in names)
            if len(self.names) != len(self.providers):
                raise ValueError("provider names must match provider count")
        if primary_index < 0 or primary_index >= len(self.providers):
            raise ValueError("primary_index is outside provider range")
        self.primary_index = int(primary_index)
        self.last_metadata: dict[str, Any] = {}
        self._collection_counter = 0

    def iter_ik_candidates(
        self,
        *,
        target_pose: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        seed_labels: tuple[str, ...],
        options: dict[str, Any],
    ) -> list[IKCandidate]:
        start_time = time.perf_counter()
        allow_parallel = bool(options.get("allow_parallel", False))
        max_count = max(1, int(options.get("max_candidate_count", 32)))
        budget_ms = _optional_positive_float(options.get("provider_time_budget_ms"))
        external_deadline = _optional_deadline(options.get("provider_deadline_monotonic"))
        provider_indices = self._provider_indices_for_options(options)
        if (
            budget_ms is not None
            and not allow_parallel
            and len(provider_indices) > 1
        ):
            provider_indices = _rotate_tuple(provider_indices, self._collection_counter)
            self._collection_counter += 1
        budget_deadline = None if budget_ms is None else start_time + budget_ms / 1000.0
        deadlines = [value for value in (budget_deadline, external_deadline) if value is not None]
        deadline = min(deadlines) if deadlines else None
        effective_budget_ms = None if deadline is None else max(0.0, (float(deadline) - start_time) * 1000.0)
        results, budget_skipped = (
            self._collect_parallel(provider_indices, target_pose, seeds, seed_labels, options, deadline)
            if allow_parallel and len(provider_indices) > 1
            else self._collect_serial(provider_indices, target_pose, seeds, seed_labels, options, deadline)
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        budget_exceeded = effective_budget_ms is not None and elapsed_ms > effective_budget_ms
        errors = [result for result in results if result.error is not None]
        self.last_metadata = {
            "provider_count": len(self.providers),
            "provider_names": list(self.names),
            "requested_provider_names": [self.names[index] for index in provider_indices],
            "provider_elapsed_ms": elapsed_ms,
            "provider_time_budget_ms": budget_ms,
            "provider_effective_time_budget_ms": effective_budget_ms,
            "provider_external_deadline_active": bool(external_deadline is not None),
            "provider_time_budget_exceeded": bool(budget_exceeded),
            "provider_time_budget_enforced": bool(deadline is not None),
            "provider_budget_skipped_names": list(budget_skipped),
            "provider_errors": [
                {"provider": result.name, "error": result.error}
                for result in errors
            ],
        }

        deduped: list[IKCandidate] = []
        seen: list[np.ndarray] = []
        sorted_results = sorted(results, key=lambda item: item.index)
        candidate_offsets = [0 for _ in sorted_results]
        while len(deduped) < max_count:
            emitted_this_round = False
            for result_index, result in enumerate(sorted_results):
                offset = candidate_offsets[result_index]
                if offset >= len(result.candidates):
                    continue
                candidate_offsets[result_index] += 1
                candidate = result.candidates[offset]
                qpos = np.asarray(candidate.qpos, dtype=float).reshape(-1)
                if any(np.allclose(qpos, existing, atol=1e-8) for existing in seen):
                    emitted_this_round = True
                    continue
                seen.append(qpos.copy())
                metadata = {
                    **dict(candidate.metadata),
                    "composite_provider_name": result.name,
                    "composite_provider_index": result.index,
                    "composite_provider_elapsed_ms": result.elapsed_ms,
                    "composite_provider_count": len(self.providers),
                }
                if budget_ms is not None:
                    metadata["provider_time_budget_ms"] = budget_ms
                    metadata["provider_elapsed_ms"] = elapsed_ms
                    metadata["provider_time_budget_exceeded"] = bool(budget_exceeded)
                    metadata["provider_time_budget_enforced"] = True
                    if budget_skipped:
                        metadata["provider_budget_skipped_names"] = list(budget_skipped)
                    if budget_exceeded:
                        metadata["provider_warning"] = "ik_provider_time_budget_exceeded"
                deduped.append(
                    IKCandidate(
                        qpos=qpos.copy(),
                        source=candidate.source,
                        seed_source=candidate.seed_source,
                        method=candidate.method,
                        metadata=metadata,
                    )
                )
                emitted_this_round = True
                if len(deduped) >= max_count:
                    return deduped
            if not emitted_this_round:
                break
        return deduped

    def _provider_indices_for_options(self, options: dict[str, Any]) -> tuple[int, ...]:
        allowlist = _string_set(options.get("provider_name_allowlist"))
        blocklist = _string_set(options.get("provider_name_blocklist"))
        indices: list[int] = []
        for index, name in enumerate(self.names):
            if allowlist is not None and name not in allowlist:
                continue
            if blocklist is not None and name in blocklist:
                continue
            indices.append(index)
        return tuple(indices)

    def forward_kinematics(self, qpos: np.ndarray) -> np.ndarray | None:
        for provider in self._providers_primary_first():
            try:
                pose = provider.forward_kinematics(qpos)
            except Exception:
                continue
            if pose is not None:
                return pose
        return None

    def analyze_singularity(self, qpos: np.ndarray) -> SingularityReport:
        for provider in self._providers_primary_first():
            try:
                report = provider.analyze_singularity(qpos)
            except Exception as exc:  # noqa: BLE001 - diagnostics only.
                report = SingularityReport(
                    status_code=None,
                    label="unavailable",
                    metric=None,
                    metadata={"error": f"{type(exc).__name__}: {exc}"},
                )
            if report.status_code is not None or report.label != "unavailable":
                return report
        return SingularityReport(status_code=None, label="unavailable", metric=None)

    def joint_position_limits(self) -> tuple[np.ndarray, np.ndarray] | None:
        for provider in self._providers_primary_first():
            getter = getattr(provider, "joint_position_limits", None)
            if getter is None:
                continue
            try:
                limits = getter()
            except Exception:
                continue
            if limits is None:
                continue
            q_min, q_max = limits
            q_min_array = np.asarray(q_min, dtype=float).reshape(-1)
            q_max_array = np.asarray(q_max, dtype=float).reshape(-1)
            if (
                q_min_array.shape == q_max_array.shape
                and np.all(np.isfinite(q_min_array))
                and np.all(np.isfinite(q_max_array))
            ):
                return q_min_array.copy(), q_max_array.copy()
        return None

    def _providers_primary_first(self) -> tuple[KinematicsProvider, ...]:
        indexed = list(enumerate(self.providers))
        indexed.sort(key=lambda item: 0 if item[0] == self.primary_index else 1 + item[0])
        return tuple(provider for _, provider in indexed)

    def _collect_serial(
        self,
        provider_indices: tuple[int, ...],
        target_pose: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        seed_labels: tuple[str, ...],
        options: dict[str, Any],
        deadline: float | None,
    ) -> tuple[list[_ProviderResult], list[str]]:
        results: list[_ProviderResult] = []
        skipped: list[str] = []
        for offset, index in enumerate(provider_indices):
            if deadline is not None and time.perf_counter() >= deadline:
                skipped.extend(self.names[skip_index] for skip_index in provider_indices[offset:])
                break
            provider = self.providers[index]
            results.append(
                self._call_provider(index, provider, target_pose, seeds, seed_labels, options, deadline)
            )
        return results, skipped

    def _collect_parallel(
        self,
        provider_indices: tuple[int, ...],
        target_pose: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        seed_labels: tuple[str, ...],
        options: dict[str, Any],
        deadline: float | None,
    ) -> tuple[list[_ProviderResult], list[str]]:
        results: list[_ProviderResult] = []
        with ThreadPoolExecutor(max_workers=len(provider_indices)) as executor:
            futures = [
                executor.submit(
                    self._call_provider,
                    index,
                    self.providers[index],
                    target_pose,
                    seeds,
                    seed_labels,
                    options,
                    deadline,
                )
                for index in provider_indices
            ]
            for future in as_completed(futures):
                results.append(future.result())
        return results, []

    def _call_provider(
        self,
        index: int,
        provider: KinematicsProvider,
        target_pose: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        seed_labels: tuple[str, ...],
        options: dict[str, Any],
        deadline: float | None,
    ) -> _ProviderResult:
        start_time = time.perf_counter()
        name = self.names[index]
        provider_options = {**options, "composite_provider_name": name}
        if deadline is not None:
            provider_options["provider_deadline_monotonic"] = float(deadline)
            provider_options["provider_remaining_time_budget_ms"] = max(
                0.0,
                (float(deadline) - time.perf_counter()) * 1000.0,
            )
        try:
            candidates = provider.iter_ik_candidates(
                target_pose=target_pose,
                seeds=seeds,
                seed_labels=seed_labels,
                options=provider_options,
            )
            return _ProviderResult(
                index=index,
                name=name,
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
                candidates=list(candidates),
            )
        except Exception as exc:  # noqa: BLE001 - one provider must not mask all others.
            return _ProviderResult(
                index=index,
                name=name,
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
                candidates=[],
                error=f"{type(exc).__name__}: {exc}",
            )


def _optional_positive_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0.0:
        return None
    return number


def _optional_deadline(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0.0:
        return None
    return number


def _string_set(value: object) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        try:
            values = [str(part).strip() for part in value]  # type: ignore[arg-type]
        except TypeError:
            return None
    out = {part for part in values if part}
    return out if out else None


def _rotate_tuple(values: tuple[int, ...], counter: int) -> tuple[int, ...]:
    if len(values) <= 1:
        return values
    offset = int(counter) % len(values)
    return values[offset:] + values[:offset]
