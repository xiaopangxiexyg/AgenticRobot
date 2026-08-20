"""Thin RealTeleop admission adapter for idle tactile lossless compression."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from collection_runtime.idle_maintenance import (
    IdleMaintenanceConfig,
    IdleMaintenanceProcessController,
    IdleMaintenanceQuiesceReceipt,
    IdleMaintenanceSnapshot,
)
from collection_runtime.tactile_recompression import (
    run_idle_tactile_recompression_worker,
)
from collection_runtime.tactile_bundle_recompression import (
    run_idle_native_tactile_recompression_worker,
)
from dataset_interface.tactile_bundle import (
    attach_tactile_bundle,
)
from dataset_interface.modality_artifacts import (
    assert_dataset_capture_integrity_compatible,
    assert_dataset_capture_writable,
)


def _catalog_revision(dataset_root: Path) -> str | None:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return None
    stat = episodes_path.stat()
    if stat.st_size <= 0:
        return None
    payload = (
        f"{dataset_root}:{stat.st_dev}:{stat.st_ino}:"
        f"{stat.st_size}:{stat.st_mtime_ns}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_capture_artifact_graph(
    *,
    dataset_root: str | Path | None,
    tactile_bundle_root: str | Path | None,
    integrity_enabled: bool,
    failure_context: str,
) -> None:
    """Validate the post-worker graph before a hot runtime transition.

    This is deliberately read-only.  The idle worker already owns all
    recompression mutations; the transition boundary only proves that the
    committed graph remains append-compatible (and, when requested, that
    stored payload hashes still verify).
    """

    if dataset_root is None:
        raise RuntimeError(
            f"{failure_context} but no dataset root is available for "
            "capture-admission validation"
        )
    admission_root = dataset_root
    root = Path(dataset_root).expanduser().resolve()
    native_identity = (
        root / "meta/tactile_bundle_v1/dataset_identity.cjson"
    ).is_file()
    if native_identity:
        if tactile_bundle_root is None or not str(tactile_bundle_root).strip():
            raise RuntimeError(
                f"{failure_context} but no tactile bundle root is available "
                "for attachment validation"
            )
        assert_dataset_capture_writable(
            admission_root,
            tactile_bundle_root=tactile_bundle_root,
        )
        attach_tactile_bundle(
            admission_root,
            tactile_bundle_root,
            verify_payload_hashes=bool(integrity_enabled),
        )
        return
    # Preserve the legacy single-root admission call shape for existing
    # callers/tests and for datasets that do not use native separation.
    assert_dataset_capture_writable(admission_root)
    if bool(integrity_enabled):
        assert_dataset_capture_integrity_compatible(admission_root)


def _is_historical_payload_hash_mismatch(error: BaseException) -> bool:
    """Recognize only a physical payload hash fault in the committed graph.

    Identity, manifest, component, and commit mismatches remain hard admission
    failures.  A payload hash mismatch is handled separately because an old
    compressed payload may be irrecoverable in-place after its raw predecessor
    has been retired; repeatedly blocking the live operator cannot repair it.
    """

    marker = "tactile artifact hash mismatch:"
    payload_marker = "tactile payload hash mismatch:"
    pending: list[BaseException | None] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current)
        if payload_marker in message:
            return True
        if marker in message:
            # Older bundle validators used one marker for both manifest and
            # payload hashes. Preserve compatibility with those errors while
            # keeping a manifest hash failure fail-closed: native manifests
            # are committed JSON files, whereas stream payloads are binary
            # artifacts (for example ``*.uint8.bin.tdelta8.zst``).
            suffix = message.split(marker, 1)[1].strip().split(None, 1)[0]
            if suffix and not suffix.lower().endswith((".json", ".cjson")):
                return True
        pending.append(current.__cause__)
        pending.append(current.__context__)
    return False


def assert_idle_tactile_transition_safe(
    receipt: IdleMaintenanceQuiesceReceipt,
    *,
    dataset_root: str | Path | None,
    tactile_bundle_root: str | Path | None = None,
    integrity_enabled: bool,
) -> None:
    """Fail closed before returning from idle mutation to a hot runtime.

    A cooperative worker error can be tolerated only after the canonical
    dataset admission checks prove that the currently referenced graph is
    complete.  A forced termination takes the same read-only recovery path:
    the graph must validate before the hot transition is admitted.  Once that
    proof succeeds the controller latches the attempted catalog revision and
    will not restart it until the catalog changes.
    """

    if not receipt.quiesced:
        raise RuntimeError(
            f"idle tactile compression did not quiesce: {receipt.reason}"
        )
    if receipt.forced_termination:
        try:
            _validate_capture_artifact_graph(
                dataset_root=dataset_root,
                tactile_bundle_root=tactile_bundle_root,
                integrity_enabled=integrity_enabled,
                failure_context=(
                    "idle tactile compression required forced termination; "
                    "artifact recovery validation failed"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - preserve Hold fail-closed path.
            if isinstance(exc, RuntimeError) and str(exc).startswith(
                "idle tactile compression required forced termination;"
            ):
                raise
            raise RuntimeError(
                "idle tactile compression required forced termination; "
                f"artifact recovery validation failed: {type(exc).__name__}: {exc}"
            ) from exc
        return
    if not receipt.worker_error:
        return
    _validate_capture_artifact_graph(
        dataset_root=dataset_root,
        tactile_bundle_root=tactile_bundle_root,
        integrity_enabled=integrity_enabled,
        failure_context="idle tactile compression reported an error",
    )


class _IdleTactileWorker:
    """Picklable worker adapter carrying the optional native bundle root."""

    def __init__(
        self,
        tactile_bundle_root: str | Path | None,
        *,
        native_min_free_bytes: int | None = None,
    ) -> None:
        value = str(tactile_bundle_root or "").strip()
        self.tactile_bundle_root = value or None
        self.native_min_free_bytes = (
            None
            if native_min_free_bytes is None
            else int(native_min_free_bytes)
        )

    def __call__(self, dataset_root: str, stop_event: Any) -> dict[str, Any]:
        if self.tactile_bundle_root is None:
            return dict(run_idle_tactile_recompression_worker(dataset_root, stop_event))
        return dict(
            run_idle_native_tactile_recompression_worker(
                dataset_root,
                stop_event,
                self.tactile_bundle_root,
                self.native_min_free_bytes,
            )
        )


class RealTeleopIdleTactileCompression:
    """Map authoritative RealTeleop state onto a generic maintenance process."""

    def __init__(
        self,
        *,
        enabled: bool,
        tactile_bundle_root: str | Path | None = None,
        tactile_bundle_min_free_bytes: int | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.tactile_bundle_root = (
            None
            if not str(tactile_bundle_root or "").strip()
            else str(Path(tactile_bundle_root).expanduser())
        )
        self._last_runtime_status: tuple[object, ...] | None = None
        worker = (
            run_idle_tactile_recompression_worker
            if self.tactile_bundle_root is None
            else _IdleTactileWorker(
                self.tactile_bundle_root,
                native_min_free_bytes=tactile_bundle_min_free_bytes,
            )
        )
        self.controller = IdleMaintenanceProcessController(
            worker,
            config=IdleMaintenanceConfig(
                process_name="real_teleop_idle_tactile_compression",
                worker_nice=19,
                stop_timeout_sec=2.0,
                terminate_timeout_sec=1.0,
            ),
            enabled=self.enabled,
        )

    def tick_runtime(
        self,
        *,
        dataset_recorder: Any | None,
        teleop_started: bool,
        hold_active: bool,
        episode_countdown_active: bool,
        startup_and_devices_ready: bool,
        log: Callable[..., None],
    ) -> IdleMaintenanceSnapshot | None:
        """Tick optional idle work and emit only changed runtime status.

        An admission/probe failure is intentionally isolated from formal
        collection and teleoperation.  Quiesce validation remains fail closed
        in :meth:`require_safe_transition`.
        """

        try:
            snapshot = self.tick(
                dataset_recorder=dataset_recorder,
                teleop_started=teleop_started,
                hold_active=hold_active,
                episode_countdown_active=episode_countdown_active,
                startup_and_devices_ready=startup_and_devices_ready,
            )
        except Exception as exc:  # noqa: BLE001 - optional idle work stays isolated.
            signature = ("tick_error", type(exc).__name__, str(exc))
            if signature != self._last_runtime_status:
                self._last_runtime_status = signature
                log(
                    "TactileCompression",
                    "idle worker admission failed; formal collection/teleop "
                    f"remain available: {type(exc).__name__}: {exc}",
                )
            return None
        signature = (
            snapshot.phase,
            snapshot.dataset_root,
            snapshot.revision,
            snapshot.outcome,
            snapshot.error,
        )
        if signature != self._last_runtime_status:
            self._last_runtime_status = signature
            log(
                "TactileCompression",
                "idle worker "
                f"phase={snapshot.phase} outcome={snapshot.outcome or '-'} "
                f"pid={snapshot.pid or '-'} root={snapshot.dataset_root or '-'}"
                + (f" error={snapshot.error}" if snapshot.error else ""),
            )
        return snapshot

    def tick(
        self,
        *,
        dataset_recorder: Any | None,
        teleop_started: bool,
        hold_active: bool,
        episode_countdown_active: bool,
        startup_and_devices_ready: bool,
        shutdown_requested: bool = False,
    ) -> IdleMaintenanceSnapshot:
        snapshot = self.controller.poll()
        recorder_idle = bool(
            dataset_recorder is not None
            and not bool(getattr(dataset_recorder, "is_recording", False))
            and not bool(getattr(dataset_recorder, "finalizing", False))
        )
        eligible = bool(
            self.enabled
            and startup_and_devices_ready
            and (not teleop_started or hold_active)
            and not episode_countdown_active
            and recorder_idle
            and not shutdown_requested
        )
        if not eligible:
            self.controller.request_quiesce(reason="real_teleop_not_idle")
            return self.controller.snapshot()
        root = Path(dataset_recorder.dataset_root).expanduser().resolve()
        revision = _catalog_revision(root)
        if revision is None:
            return self.controller.snapshot()
        if snapshot.dataset_root not in {None, str(root)} and self.controller.running:
            self.controller.request_quiesce(reason="dataset_root_changed")
            return self.controller.snapshot()
        self.controller.start_if_eligible(root, revision=revision)
        return self.controller.snapshot()

    def begin_latency_sensitive_transition(self, *, reason: str) -> None:
        self.controller.request_quiesce(reason=reason)

    def require_quiesced(
        self,
        *,
        reason: str,
        timeout_sec: float | None = None,
    ) -> IdleMaintenanceQuiesceReceipt:
        return self.controller.quiesce(reason=reason, timeout_sec=timeout_sec)

    def acknowledge_safe_transition(
        self,
        receipt: IdleMaintenanceQuiesceReceipt,
    ) -> None:
        """Consume a sticky worker result after artifact admission succeeds."""

        if receipt.forced_termination:
            self.controller.acknowledge_forced_quiesce_receipt(receipt)
        else:
            self.controller.acknowledge_quiesce_receipt(receipt)

    def require_safe_transition(
        self,
        *,
        reason: str,
        dataset_root: str | Path | None,
        tactile_bundle_root: str | Path | None = None,
        integrity_enabled: bool,
        log: Callable[..., None],
        timeout_sec: float | None = None,
    ) -> IdleMaintenanceQuiesceReceipt:
        """Quiesce, validate the referenced dataset, then consume the receipt."""

        receipt = self.require_quiesced(
            reason=reason,
            timeout_sec=timeout_sec,
        )
        try:
            assert_idle_tactile_transition_safe(
                receipt,
                dataset_root=dataset_root,
                tactile_bundle_root=(
                    self.tactile_bundle_root
                    if tactile_bundle_root is None
                    else tactile_bundle_root
                ),
                integrity_enabled=integrity_enabled,
            )
        except Exception as exc:  # noqa: BLE001 - classify one explicit degraded path.
            if not _is_historical_payload_hash_mismatch(exc):
                raise
            degraded_reason = (
                "historical tactile payload hash mismatch; idle compression disabled"
            )
            self.controller.disable_after_degraded_quiesce_receipt(
                receipt,
                reason=degraded_reason,
            )
            self.enabled = False
            log(
                "TactileCompression",
                "历史触觉 payload 校验失败，已保留原文件并停用后台压缩；"
                "本次 Home/数采继续，需离线修复该 episode 后重启 idle 压缩："
                f"{type(exc).__name__}: {exc}",
            )
            return receipt
        self.acknowledge_safe_transition(receipt)
        log(
            "TactileCompression",
            "quiesced before latency-sensitive transition "
            f"reason={reason} forced={receipt.forced_termination} "
            f"syncfs={receipt.filesystem_barrier_applied} "
            f"recovery={'validated' if receipt.forced_termination else 'not_needed'}",
        )
        return receipt

    def snapshot(self) -> IdleMaintenanceSnapshot:
        return self.controller.poll()

    def close(self) -> None:
        self.controller.close()


__all__ = [
    "RealTeleopIdleTactileCompression",
    "assert_idle_tactile_transition_safe",
]
