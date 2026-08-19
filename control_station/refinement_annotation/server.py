"""Loopback-only HTTP service for fine-stage annotation.

The handler owns presentation and request validation only.  Dataset discovery,
draft persistence, immutable publication, and training-index validation remain
in their lower-level owners.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Sequence
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from collection_runtime.progress.stage_annotation_store import (
    StageAnnotationStoreError,
)
from collection_runtime.progress.detail_stage_annotation_store import (
    DetailStageAnnotationStoreError,
)

from .application import (
    RefinementAnnotationApplication,
    RefinementAnnotationApplicationError,
)
from .media import BrowserMediaCache, BrowserMediaError


_STATIC_ROOT = Path(__file__).with_name("static")
_STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
}
_LOCAL_BIND_HOSTS = frozenset({"127.0.0.1", "localhost"})
_MAX_JSON_BODY_BYTES = 4 * 1024 * 1024
_RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")
_OPERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)


class _OperationProgressRegistry:
    """Small in-memory registry for live browser operation progress."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, dict[str, Any]] = {}

    def begin(self, operation_id: str, *, kind: str, message: str) -> None:
        now = time.time()
        with self._lock:
            self._values[operation_id] = {
                "operation_id": operation_id,
                "kind": kind,
                "percent": 0,
                "message": message,
                "active": True,
                "error": None,
                "started_at": now,
                "updated_at": now,
            }
            if len(self._values) > 64:
                oldest = min(
                    self._values,
                    key=lambda key: self._values[key]["updated_at"],
                )
                if oldest != operation_id:
                    self._values.pop(oldest, None)

    def update(self, operation_id: str, percent: int, message: str) -> None:
        with self._lock:
            value = self._values.get(operation_id)
            if value is None or not value["active"]:
                return
            value["percent"] = max(0, min(99, int(percent)))
            value["message"] = str(message)
            value["updated_at"] = time.time()

    def finish(self, operation_id: str, message: str) -> None:
        with self._lock:
            value = self._values.get(operation_id)
            if value is None:
                return
            value.update(
                percent=100,
                message=str(message),
                active=False,
                updated_at=time.time(),
            )

    def fail(self, operation_id: str, error: str) -> None:
        with self._lock:
            value = self._values.get(operation_id)
            if value is None:
                return
            value.update(
                message="操作失败",
                active=False,
                error=str(error),
                updated_at=time.time(),
            )

    def snapshot(self, operation_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._values.get(operation_id)
            return None if value is None else dict(value)


class _AnnotationHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        dataset_root: str | Path | None,
        annotator_id: str,
        second_review: bool,
        allowed_source_roots: Sequence[str | Path],
    ) -> None:
        self._application_lock = threading.RLock()
        self._media_cache = BrowserMediaCache()
        self._operation_progress = _OperationProgressRegistry()
        self._annotation_application: (
            RefinementAnnotationApplication | None
        ) = None
        self._suggested_dataset_root = (
            ""
            if dataset_root is None
            else str(Path(dataset_root).expanduser().resolve())
        )
        self._suggested_annotator_id = str(annotator_id).strip()
        self._suggested_second_review = bool(second_review)
        self._allowed_source_roots = tuple(
            Path(value).expanduser().resolve()
            for value in allowed_source_roots
        )
        super().__init__(server_address, handler_class)
        try:
            if self._suggested_dataset_root and self._suggested_annotator_id:
                self.open_workspace(
                    dataset_root=self._suggested_dataset_root,
                    annotator_id=self._suggested_annotator_id,
                    second_review=self._suggested_second_review,
                )
        except Exception:
            self._media_cache.close()
            self.server_close()
            raise

    def workspace_payload(self) -> dict[str, Any]:
        with self._application_lock:
            application = self._annotation_application
            return {
                "configured": application is not None,
                "dataset_root": (
                    str(application.dataset_root)
                    if application is not None
                    else self._suggested_dataset_root
                ),
                "annotator_id": (
                    application.annotator_id
                    if application is not None
                    else self._suggested_annotator_id
                ),
                "second_review": (
                    application.annotation_mode == "SECOND_REVIEW"
                    if application is not None
                    else self._suggested_second_review
                ),
                "annotation_mode": (
                    application.annotation_mode
                    if application is not None
                    else (
                        "SECOND_REVIEW"
                        if self._suggested_second_review
                        else "STANDARD"
                    )
                ),
            }

    def require_application(self) -> RefinementAnnotationApplication:
        with self._application_lock:
            application = self._annotation_application
            if application is None:
                raise RefinementAnnotationApplicationError(
                    "请先在 GUI 中选择数据集目录并填写标注员 ID。"
                )
            return application

    def _validated_workspace_inputs(
        self,
        *,
        dataset_root: object,
        annotator_id: object,
        second_review: object = False,
    ) -> tuple[Path, str]:
        dataset_text = str(dataset_root or "").strip()
        annotator_text = str(annotator_id or "").strip()
        if not dataset_text:
            raise RefinementAnnotationApplicationError(
                "数据集绝对路径不能为空。"
            )
        candidate = Path(dataset_text).expanduser()
        if not candidate.is_absolute():
            raise RefinementAnnotationApplicationError(
                "数据集目录必须使用绝对路径。"
            )
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise RefinementAnnotationApplicationError(
                f"数据集目录不存在或不是文件夹：{candidate}"
            )
        if not annotator_text:
            raise RefinementAnnotationApplicationError(
                "标注员 ID 不能为空。"
            )
        return candidate, annotator_text

    def _workspace_result(
        self,
        application: RefinementAnnotationApplication,
    ) -> dict[str, Any]:
        episode_snapshot = application.list_episodes(
            wait_for_refresh=False,
            include_annotation_summaries=False,
        )
        return {
            **self.workspace_payload(),
            "eligible_episode_count": len(episode_snapshot["episodes"]),
            "episode_scan_in_progress": episode_snapshot["refreshing"],
            "episode_scan_error": episode_snapshot["refresh_error"],
        }

    def open_workspace(
        self,
        *,
        dataset_root: object,
        annotator_id: object,
        second_review: object = False,
    ) -> dict[str, Any]:
        candidate, annotator_text = self._validated_workspace_inputs(
            dataset_root=dataset_root,
            annotator_id=annotator_id,
        )
        with self._application_lock:
            current = self._annotation_application
            if current is not None:
                if (
                    current.dataset_root == candidate
                    and current.annotator_id == annotator_text
                    and current.annotation_mode
                    == (
                        "SECOND_REVIEW"
                        if bool(second_review)
                        else "STANDARD"
                    )
                ):
                    return self._workspace_result(current)
                raise RefinementAnnotationApplicationError(
                    "当前服务已经打开一个标注工作区。请使用页面上的“更换数据集”按钮切换，"
                    "不要重复提交打开请求。"
                )
            application = RefinementAnnotationApplication(
                candidate,
                annotator_id=annotator_text,
                second_review=bool(second_review),
                allowed_source_roots=self._allowed_source_roots,
                defer_initial_refresh=True,
            )
            self._annotation_application = application
            self._suggested_dataset_root = str(candidate)
            self._suggested_annotator_id = annotator_text
            self._suggested_second_review = bool(second_review)
            return self._workspace_result(application)

    def switch_workspace(
        self,
        *,
        dataset_root: object,
        annotator_id: object,
        second_review: object = False,
    ) -> dict[str, Any]:
        """Switch datasets without dropping the old lock on target failure."""

        candidate, annotator_text = self._validated_workspace_inputs(
            dataset_root=dataset_root,
            annotator_id=annotator_id,
        )
        old_application: RefinementAnnotationApplication | None = None
        with self._application_lock:
            current = self._annotation_application
            if current is None:
                application = RefinementAnnotationApplication(
                    candidate,
                    annotator_id=annotator_text,
                    second_review=bool(second_review),
                    allowed_source_roots=self._allowed_source_roots,
                    defer_initial_refresh=True,
                )
                self._annotation_application = application
            elif (
                current.dataset_root == candidate
                and current.annotator_id == annotator_text
                and current.annotation_mode
                == (
                    "SECOND_REVIEW"
                    if bool(second_review)
                    else "STANDARD"
                )
            ):
                return self._workspace_result(current)
            elif current.dataset_root == candidate:
                raise RefinementAnnotationApplicationError(
                    "目标目录就是当前数据集，但标注员 ID 或标注模式不同；"
                    "为避免草稿身份混淆，请关闭服务后重新启动。"
                )
            else:
                # Acquire the new dataset lock first. If this fails, the old
                # application and its lock remain usable.
                application = RefinementAnnotationApplication(
                    candidate,
                    annotator_id=annotator_text,
                    second_review=bool(second_review),
                    allowed_source_roots=self._allowed_source_roots,
                    defer_initial_refresh=True,
                )
                old_application = current
                self._annotation_application = application
            self._suggested_dataset_root = str(candidate)
            self._suggested_annotator_id = annotator_text
            self._suggested_second_review = bool(second_review)
            result = self._workspace_result(application)
        if old_application is not None:
            old_application.close()
        return result

    def close_application(self) -> None:
        with self._application_lock:
            application = self._annotation_application
            self._annotation_application = None
        if application is not None:
            application.close()

    def browse_dataset_root(self, *, path: object) -> dict[str, Any]:
        """Return a directory-only view for the browser folder picker."""

        path_text = str(path or "").strip()
        if not path_text:
            path_text = self._suggested_dataset_root or str(Path.home())
        candidate = Path(path_text).expanduser()
        if not candidate.is_absolute():
            raise RefinementAnnotationApplicationError(
                "目录浏览路径必须使用绝对路径。"
            )
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise RefinementAnnotationApplicationError(
                f"目录不存在或不是文件夹：{candidate}"
            )
        directories: list[dict[str, str]] = []
        try:
            with os.scandir(candidate) as entries:
                for entry in entries:
                    try:
                        if not entry.is_dir(follow_symlinks=True):
                            continue
                    except OSError:
                        # One inaccessible child must not hide the other usable
                        # directories in the picker.
                        continue
                    directories.append(
                        {
                            "name": entry.name,
                            # Resolve only the selected directory later. Avoid
                            # one remote canonicalization round trip per child.
                            "path": os.path.abspath(entry.path),
                        }
                    )
            directories.sort(key=lambda value: value["name"].casefold())
        except OSError as exc:
            raise RefinementAnnotationApplicationError(
                f"无法读取目录：{candidate} ({exc})"
            ) from exc
        parent = None if candidate.parent == candidate else str(candidate.parent)
        return {
            "current": str(candidate),
            "parent": parent,
            "directories": directories,
        }

    def resolve_browser_media(self, source: Path) -> Path:
        return self._media_cache.resolve(source)

    def begin_operation(
        self,
        operation_id: str,
        *,
        kind: str,
        message: str,
    ) -> None:
        self._operation_progress.begin(
            operation_id,
            kind=kind,
            message=message,
        )

    def update_operation(
        self,
        operation_id: str,
        percent: int,
        message: str,
    ) -> None:
        self._operation_progress.update(operation_id, percent, message)

    def finish_operation(self, operation_id: str, message: str) -> None:
        self._operation_progress.finish(operation_id, message)

    def fail_operation(self, operation_id: str, error: str) -> None:
        self._operation_progress.fail(operation_id, error)

    def operation_progress(self, operation_id: str) -> dict[str, Any] | None:
        return self._operation_progress.snapshot(operation_id)

    def close_media_cache(self) -> None:
        self._media_cache.close()


class _AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "DexTeleopRefinementAnnotation/1"

    @property
    def application(self) -> RefinementAnnotationApplication:
        return self.server.require_application()

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def _security_headers(self, *, cache: str = "no-store") -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; media-src 'self'; img-src 'self'; "
            "frame-ancestors 'none'",
        )
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", cache)

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        *,
        extra_headers: tuple[tuple[str, str], ...] = (),
        cache: str = "no-store",
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in extra_headers:
            self.send_header(name, value)
        self._security_headers(cache=cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
    ) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(
            status,
            encoded,
            "application/json; charset=utf-8",
        )

    def _send_error_json(
        self,
        status: HTTPStatus,
        message: str,
    ) -> None:
        try:
            self._send_json(
                status,
                {"ok": False, "error": str(message)},
            )
        except _CLIENT_DISCONNECT_ERRORS:
            # The browser may cancel a media/API request while the server is
            # preparing its response.  There is no socket left on which an
            # error response could be delivered in that case.
            return

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("application/json"):
            raise RefinementAnnotationApplicationError(
                "Request Content-Type must be application/json."
            )
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RefinementAnnotationApplicationError(
                "Invalid Content-Length."
            ) from exc
        if length <= 0 or length > _MAX_JSON_BODY_BYTES:
            raise RefinementAnnotationApplicationError(
                "JSON request body is empty or too large."
            )
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RefinementAnnotationApplicationError(
                f"Invalid JSON request: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RefinementAnnotationApplicationError(
                "JSON request root must be an object."
            )
        return value

    def _require_same_origin(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            return
        parsed = urlsplit(origin)
        expected_host = self.headers.get("Host", "")
        if parsed.scheme not in {"http", "https"} or parsed.netloc != expected_host:
            raise RefinementAnnotationApplicationError(
                "Cross-origin mutation requests are rejected."
            )

    def _operation_id(self) -> str | None:
        value = str(self.headers.get("X-Annotation-Operation", "")).strip()
        if not value:
            return None
        if _OPERATION_ID_PATTERN.fullmatch(value) is None:
            raise RefinementAnnotationApplicationError(
                "Invalid annotation operation identity."
            )
        return value

    @staticmethod
    def _api_identity(path: str, prefix: str) -> str | None:
        if not path.startswith(prefix):
            return None
        value = unquote(path[len(prefix) :]).strip()
        if not value or "/" in value or value in {".", ".."}:
            return None
        return value

    def _serve_static(self, path: str) -> bool:
        relative = _STATIC_ROUTES.get(path)
        if relative is None:
            return False
        file_path = (_STATIC_ROOT / relative).resolve()
        if file_path.parent != _STATIC_ROOT.resolve() or not file_path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "Static asset is missing.")
            return True
        payload = file_path.read_bytes()
        content_type = (
            mimetypes.guess_type(file_path.name)[0]
            or "application/octet-stream"
        )
        self._send_bytes(
            HTTPStatus.OK,
            payload,
            f"{content_type}; charset=utf-8"
            if content_type.startswith(("text/", "application/javascript"))
            else content_type,
        )
        return True

    def _serve_media(self, path: str) -> bool:
        if not path.startswith("/media/"):
            return False
        parts = tuple(unquote(value) for value in path.split("/")[2:])
        if (
            len(parts) != 2
            or any(not value or value in {".", ".."} for value in parts)
        ):
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown media path.")
            return True
        try:
            media_path = self.server.resolve_browser_media(
                self.application.media_path(parts[0], parts[1])
            )
            file_size = media_path.stat().st_size
            range_header = self.headers.get("Range")
            if not range_header:
                self.send_response(HTTPStatus.OK.value)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(media_path.name)[0] or "video/mp4",
                )
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self._security_headers(cache="private, max-age=3600")
                self.end_headers()
                if self.command != "HEAD":
                    with media_path.open("rb") as handle:
                        while chunk := handle.read(1024 * 1024):
                            self.wfile.write(chunk)
                return True
            match = _RANGE_PATTERN.fullmatch(range_header.strip())
            if match is None:
                raise RefinementAnnotationApplicationError(
                    "Only one HTTP byte range is supported."
                )
            start_text, end_text = match.groups()
            if not start_text and not end_text:
                raise RefinementAnnotationApplicationError(
                    "Invalid HTTP byte range."
                )
            if start_text:
                start = int(start_text)
                end = (
                    min(int(end_text), file_size - 1)
                    if end_text
                    else file_size - 1
                )
            else:
                suffix = int(end_text)
                if suffix <= 0:
                    raise RefinementAnnotationApplicationError(
                        "Invalid suffix byte range."
                    )
                start = max(0, file_size - suffix)
                end = file_size - 1
            if start < 0 or start >= file_size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE.value)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self._security_headers()
                self.end_headers()
                return True
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT.value)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(media_path.name)[0] or "video/mp4",
            )
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header(
                "Content-Range",
                f"bytes {start}-{end}/{file_size}",
            )
            self._security_headers(cache="private, max-age=3600")
            self.end_headers()
            if self.command != "HEAD":
                remaining = length
                with media_path.open("rb") as handle:
                    handle.seek(start)
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            return True
        except _CLIENT_DISCONNECT_ERRORS:
            # Replacing a video source, seeking, or cancelling prefetch can
            # close the browser connection while a media range is streaming.
            return True
        except (
            BrowserMediaError,
            OSError,
            RefinementAnnotationApplicationError,
            StageAnnotationStoreError,
            ValueError,
        ) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return True

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if self._serve_static(path) or self._serve_media(path):
                return
            if path == "/api/dataset-browser":
                query = parse_qs(
                    urlsplit(self.path).query,
                    keep_blank_values=True,
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        **self.server.browse_dataset_root(
                            path=query.get("path", [""])[0],
                        ),
                    },
                )
                return
            if path == "/api/workspace":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "workspace": self.server.workspace_payload(),
                    },
                )
                return
            if path == "/api/operation-progress":
                query = parse_qs(
                    urlsplit(self.path).query,
                    keep_blank_values=True,
                )
                operation_id = query.get("id", [""])[0].strip()
                if _OPERATION_ID_PATTERN.fullmatch(operation_id) is None:
                    raise RefinementAnnotationApplicationError(
                        "A valid operation progress identity is required."
                    )
                progress = self.server.operation_progress(operation_id)
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "progress": progress},
                )
                return
            if path == "/api/episodes":
                query = parse_qs(
                    urlsplit(self.path).query,
                    keep_blank_values=True,
                )
                stale_ok = query.get("stale_ok", ["0"])[0] in {
                    "1",
                    "true",
                    "yes",
                }
                force_refresh = query.get("refresh", ["0"])[0] in {
                    "1",
                    "true",
                    "yes",
                }
                page_offset: int | None = None
                page_limit: int | None = None
                has_offset = "offset" in query
                has_limit = "limit" in query
                if has_offset or has_limit:
                    if not (has_offset and has_limit):
                        raise RefinementAnnotationApplicationError(
                            "Episode page offset and limit must be supplied together."
                        )
                    try:
                        page_offset = int(query["offset"][0])
                        page_limit = int(query["limit"][0])
                    except (TypeError, ValueError) as exc:
                        raise RefinementAnnotationApplicationError(
                            "Episode page offset and limit must be integers."
                        ) from exc
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        **self.application.list_episodes(
                            wait_for_refresh=not stale_ok or force_refresh,
                            force_refresh=force_refresh,
                            include_annotation_summaries=False,
                            offset=page_offset,
                            limit=page_limit,
                        ),
                    },
                )
                return
            episode_uuid = self._api_identity(path, "/api/episode/")
            if episode_uuid is not None:
                operation_id = self._operation_id()
                if operation_id is not None:
                    self.server.begin_operation(
                        operation_id,
                        kind="OPEN_EPISODE",
                        message="正在打开 Episode",
                    )
                try:
                    if operation_id is None:
                        episode = self.application.episode_detail(episode_uuid)
                    else:
                        episode = self.application.episode_detail(
                            episode_uuid,
                            progress_callback=lambda percent, message: (
                                self.server.update_operation(
                                    operation_id,
                                    percent,
                                    message,
                                )
                            ),
                        )
                except Exception as exc:
                    if operation_id is not None:
                        self.server.fail_operation(operation_id, str(exc))
                    raise
                if operation_id is not None:
                    self.server.finish_operation(
                        operation_id,
                        "Episode 已打开",
                    )
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "episode": episode},
                )
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown route.")
        except _CLIENT_DISCONNECT_ERRORS:
            return
        except (
            OSError,
            RefinementAnnotationApplicationError,
            StageAnnotationStoreError,
            ValueError,
        ) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_PUT(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        episode_uuid = self._api_identity(path, "/api/draft/")
        if episode_uuid is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown route.")
            return
        try:
            self._require_same_origin()
            draft = self._read_json()
            saved = self.application.save_draft(episode_uuid, draft)
            self._send_json(HTTPStatus.OK, {"ok": True, "draft": saved})
        except (
            OSError,
            RefinementAnnotationApplicationError,
            StageAnnotationStoreError,
            ValueError,
        ) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/episode-summaries":
            try:
                self._require_same_origin()
                request = self._read_json()
                episode_uuids = request.get("episode_uuids", ())
                if not isinstance(episode_uuids, list):
                    raise RefinementAnnotationApplicationError(
                        "episode_uuids must be a JSON list."
                    )
                payload = self.application.episode_annotation_summaries(
                    episode_uuids
                )
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, **payload},
                )
            except (
                OSError,
                RefinementAnnotationApplicationError,
                StageAnnotationStoreError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if path == "/api/switch-workspace":
            try:
                self._require_same_origin()
                request = self._read_json()
                workspace = self.server.switch_workspace(
                    dataset_root=request.get("dataset_root"),
                    annotator_id=request.get("annotator_id"),
                    second_review=request.get("second_review", False),
                )
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "workspace": workspace},
                )
            except (
                OSError,
                RefinementAnnotationApplicationError,
                RuntimeError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if path == "/api/workspace":
            try:
                self._require_same_origin()
                request = self._read_json()
                workspace = self.server.open_workspace(
                    dataset_root=request.get("dataset_root"),
                    annotator_id=request.get("annotator_id"),
                    second_review=request.get("second_review", False),
                )
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "workspace": workspace},
                )
            except (
                OSError,
                RefinementAnnotationApplicationError,
                RuntimeError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        reopen_episode_uuid = self._api_identity(path, "/api/reopen/")
        if reopen_episode_uuid is not None:
            try:
                self._require_same_origin()
                request = self._read_json()
                reopened = self.application.reopen_sealed_annotation(
                    reopen_episode_uuid,
                    annotation_set_id=request.get("annotation_set_id"),
                )
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, **reopened},
                )
            except (
                OSError,
                RefinementAnnotationApplicationError,
                StageAnnotationStoreError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        reflow_episode_uuid = self._api_identity(
            path,
            "/api/reflow-failed-stage/",
        )
        capture_end_episode_uuid = self._api_identity(
            path,
            "/api/reflow-capture-end-step/",
        )
        if capture_end_episode_uuid is not None:
            try:
                self._require_same_origin()
                request = self._read_json()
                result = self.application.reflow_capture_end_step(
                    capture_end_episode_uuid,
                    request.get("draft", {}),
                    end_progress_step_id=request.get(
                        "end_progress_step_id",
                        "",
                    ),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, **result})
            except (
                OSError,
                RefinementAnnotationApplicationError,
                StageAnnotationStoreError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if reflow_episode_uuid is not None:
            try:
                self._require_same_origin()
                request = self._read_json()
                result = self.application.reflow_failed_attempt_stage(
                    reflow_episode_uuid,
                    request.get("draft", {}),
                    attempt_id=request.get("attempt_id", ""),
                    failed_stage_id=request.get("failed_stage_id", ""),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, **result})
            except (
                OSError,
                RefinementAnnotationApplicationError,
                StageAnnotationStoreError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        recoverable_episode_uuid = self._api_identity(
            path,
            "/api/recoverable-failure/",
        )
        if recoverable_episode_uuid is not None:
            try:
                self._require_same_origin()
                request = self._read_json()
                result = self.application.add_recoverable_failure(
                    recoverable_episode_uuid,
                    request.get("draft", {}),
                    onset_at=request.get("onset_at"),
                    recovered_at=request.get("recovered_at"),
                    note=request.get("note"),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, **result})
            except (
                OSError,
                RefinementAnnotationApplicationError,
                StageAnnotationStoreError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        reconcile_episode_uuid = self._api_identity(
            path,
            "/api/reconcile-abnormal-intervals/",
        )
        if reconcile_episode_uuid is not None:
            try:
                self._require_same_origin()
                request = self._read_json()
                result = self.application.reconcile_abnormal_intervals(
                    reconcile_episode_uuid,
                    request.get("draft", {}),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, **result})
            except (
                OSError,
                RefinementAnnotationApplicationError,
                StageAnnotationStoreError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        detail_episode_uuid = self._api_identity(
            path,
            "/api/commit-detail/",
        )
        if detail_episode_uuid is not None:
            try:
                self._require_same_origin()
                draft = self._read_json()
                operation_id = self._operation_id()
                if operation_id is not None:
                    self.server.begin_operation(
                        operation_id,
                        kind="COMMIT_DETAIL_ANNOTATION",
                        message="正在校验详细标注",
                    )
                try:
                    if operation_id is None:
                        report = self.application.commit_detail(
                            detail_episode_uuid,
                            draft,
                        )
                    else:
                        report = self.application.commit_detail(
                            detail_episode_uuid,
                            draft,
                            progress_callback=lambda percent, message: (
                                self.server.update_operation(
                                    operation_id,
                                    percent,
                                    message,
                                )
                            ),
                        )
                except Exception as exc:
                    if operation_id is not None:
                        self.server.fail_operation(operation_id, str(exc))
                    raise
                if operation_id is not None:
                    self.server.finish_operation(
                        operation_id,
                        "详细标注已封存",
                    )
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "report": report},
                )
            except (
                OSError,
                RefinementAnnotationApplicationError,
                DetailStageAnnotationStoreError,
                StageAnnotationStoreError,
                ValueError,
            ) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        episode_uuid = self._api_identity(path, "/api/commit/")
        if episode_uuid is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown route.")
            return
        try:
            self._require_same_origin()
            draft = self._read_json()
            operation_id = self._operation_id()
            if operation_id is not None:
                self.server.begin_operation(
                    operation_id,
                    kind="COMMIT_ANNOTATION",
                    message="正在校验标注",
                )
            try:
                if operation_id is None:
                    report = self.application.commit(episode_uuid, draft)
                else:
                    report = self.application.commit(
                        episode_uuid,
                        draft,
                        progress_callback=lambda percent, message: (
                            self.server.update_operation(
                                operation_id,
                                percent,
                                message,
                            )
                        ),
                    )
            except Exception as exc:
                if operation_id is not None:
                    self.server.fail_operation(operation_id, str(exc))
                raise
            if operation_id is not None:
                self.server.finish_operation(
                    operation_id,
                    "标注已封存",
                )
            self._send_json(HTTPStatus.OK, {"ok": True, "report": report})
        except (
            OSError,
            RefinementAnnotationApplicationError,
            StageAnnotationStoreError,
            ValueError,
        ) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        self._send_error_json(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "Dataset deletion is not supported.",
        )


class RefinementAnnotationGui:
    """Own the loopback server thread and one Episode-scoped editor."""

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        annotator_id: str = "",
        second_review: bool = False,
        allowed_source_roots: Sequence[str | Path] = (),
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
    ) -> None:
        normalized_host = str(host).strip()
        if normalized_host not in _LOCAL_BIND_HOSTS:
            raise ValueError(
                "Refinement annotation GUI only permits a loopback bind."
            )
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError("port must be an integer in [0, 65535].")
        if not 0 <= port <= 65535:
            raise ValueError("port must be an integer in [0, 65535].")
        self.server = _AnnotationHttpServer(
            (normalized_host, port),
            _AnnotationHandler,
            dataset_root=dataset_root,
            annotator_id=annotator_id,
            second_review=second_review,
            allowed_source_roots=allowed_source_roots,
        )
        bound_host, bound_port = self.server.server_address[:2]
        visible_host = (
            "127.0.0.1" if str(bound_host) in {"0.0.0.0", "::"} else bound_host
        )
        self.url = f"http://{visible_host}:{bound_port}/"
        self.open_browser = bool(open_browser)
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        if self._thread is not None:
            return self.url
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            name="refinement-annotation-gui",
            daemon=True,
        )
        self._thread.start()
        if self.open_browser:
            webbrowser.open(self.url)
        return self.url

    def close(self) -> None:
        if self._thread is not None:
            self.server.shutdown()
            self._thread.join(timeout=5.0)
            self._thread = None
        self.server.close_media_cache()
        self.server.server_close()
        self.server.close_application()

    def __enter__(self) -> "RefinementAnnotationGui":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["RefinementAnnotationGui"]
