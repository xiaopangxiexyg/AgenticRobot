"""One-shot request and background-write lifecycle for Plus curation."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from typing import Any

from ..collection.episode_curation import EpisodeCurationSession


StateCallback = Callable[[], None]
MessageCallback = Callable[[str], None]
SuccessCallback = Callable[[Any], None]


class RealTeleopPlusCurationRuntime:
    """Own the pending application request and its single writer thread.

    ``EpisodeCurationSession`` remains the only persistence/identity owner.
    This adapter only serializes one GUI request into one background call and
    reports state changes through injected callbacks.
    """

    def __init__(self, session: EpisodeCurationSession) -> None:
        self.session = session
        self._pending_request: dict[str, object] | None = None
        self._save_thread: threading.Thread | None = None

    def submit_request_text(
        self,
        text: str,
        *,
        on_state_changed: StateCallback,
    ) -> bool:
        """Decode and admit one application request for control-loop pickup."""

        try:
            payload = json.loads(str(text or "{}"))
            if not isinstance(payload, Mapping):
                raise ValueError("Plus curation payload must be an object")
            request = {
                "episode_index": int(payload.get("episode_index")),
                "pending_token": str(
                    payload.get("pending_token", "") or ""
                ).strip(),
                "disposition": str(payload.get("disposition", "unknown")),
                "reason_code": str(payload.get("reason_code", "unknown")),
            }
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.session.status = "error"
            self.session.message = f"Plus 整理参数无效: {exc}"
            on_state_changed()
            return False

        self._pending_request = request
        self.session.status = "saving"
        self.session.message = "Plus 整理提交已接收。"
        on_state_changed()
        return True

    def consume_request(self) -> dict[str, object] | None:
        request = self._pending_request
        self._pending_request = None
        return request

    def save_in_flight(self) -> bool:
        return bool(
            self.session.enabled
            and (
                self.session.status == "saving"
                or (
                    self._save_thread is not None
                    and self._save_thread.is_alive()
                )
            )
        )

    def start_save(
        self,
        *,
        episode_index: int,
        pending_token: str,
        disposition: str,
        reason_code: str,
        on_state_changed: StateCallback,
        on_failure: MessageCallback,
        on_success: SuccessCallback,
    ) -> threading.Thread | None:
        """Start at most one save and preserve the snapshotted source identity."""

        if not self.session.enabled:
            return None
        if self._save_thread is not None and self._save_thread.is_alive():
            return None
        if not self.session.matches_pending(
            episode_index=int(episode_index),
            pending_token=str(pending_token),
        ):
            self.session.status = "error"
            self.session.message = (
                "Plus 整理请求已过期：当前待整理 episode 已变化。"
            )
            on_state_changed()
            return None
        try:
            self.session.set_review(
                disposition=str(disposition),
                reason_code=str(reason_code),
            )
        except ValueError as exc:
            self.session.status = "error"
            self.session.message = f"Plus 整理参数无效: {exc}"
            on_state_changed()
            return None

        request_episode_index = self.session.pending_episode_index
        request_dataset_root = self.session.pending_dataset_root
        request_pending_token = self.session.pending_token
        request_variant_id = self.session.variant_id
        self.session.status = "saving"
        self.session.message = "Plus sidecar 正在后台写入。"
        on_state_changed()

        def write_sidecar() -> None:
            try:
                record = self.session.save(
                    disposition=str(disposition),
                    reason_code=str(reason_code),
                    pending_token=request_pending_token,
                    variant_id=request_variant_id,
                    source_episode_index=request_episode_index,
                    source_dataset_root=request_dataset_root,
                )
            except Exception as exc:  # noqa: BLE001 - optional overlay is non-fatal.
                message = f"Plus sidecar 写入失败: {type(exc).__name__}: {exc}"
                if (
                    self.session.pending_episode_index == request_episode_index
                    and self.session.pending_dataset_root == request_dataset_root
                    and self.session.pending_token == request_pending_token
                ):
                    self.session.status = "error"
                    self.session.message = message
                on_failure(message)
            else:
                on_success(record)
            finally:
                on_state_changed()

        self._save_thread = threading.Thread(
            target=write_sidecar,
            name="plus-episode-curation-save",
            daemon=True,
        )
        self._save_thread.start()
        return self._save_thread


__all__ = ["RealTeleopPlusCurationRuntime"]
