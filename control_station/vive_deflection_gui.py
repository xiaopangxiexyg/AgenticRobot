"""Loopback GUI for the standalone read-only VIVE deflection viewer."""

from __future__ import annotations

import json
import secrets
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .relative_deflection_view import (
    RELATIVE_DEFLECTION_VIEW_CSS,
    RELATIVE_DEFLECTION_VIEW_HTML,
    RELATIVE_DEFLECTION_VIEW_JS,
)
from .trajectory_guidance_view import (
    TRAJECTORY_GUIDANCE_VIEW_CSS,
    TRAJECTORY_GUIDANCE_VIEW_HTML,
    TRAJECTORY_GUIDANCE_VIEW_JS,
)
from .vive_deflection import ViveDeflectionController


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})

_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VIVE 偏转查看器</title>
<style nonce="__CSP_NONCE__">
:root { color-scheme:dark; --bg:#08101c; --panel:#111827; --line:#334155; --text:#e5edf8; --muted:#93a4ba; --blue:#73b7ff; --shadow:0 18px 46px rgba(0,0,0,.3); }
* { box-sizing:border-box; }
body { margin:0; min-height:100vh; background:radial-gradient(circle at top,#17253b,var(--bg) 62%); color:var(--text); font-family:Inter,system-ui,sans-serif; }
main { width:min(1500px,calc(100% - 28px)); margin:0 auto; padding:30px 0; }
h1,h2,p { margin-top:0; } h1 { margin-bottom:8px; } h2 { font-size:16px; }
.subtitle,.subtle { color:var(--muted); }
.toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:18px 0; }
button { min-height:40px; padding:8px 14px; border:1px solid var(--line); border-radius:9px; color:var(--text); background:#14233a; cursor:pointer; }
button:hover { border-color:var(--blue); }
.calibration-status { min-width:220px; }
.axis-warning { margin:-8px 0 16px; padding:9px 11px; border:1px solid #8a6530; border-radius:8px; color:#ffd27a; background:rgba(138,101,48,.14); }
.label { color:var(--blue); }
.card-head { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; }
.viewer-workspace { display:grid; grid-template-columns:minmax(440px,.88fr) minmax(560px,1.12fr); gap:16px; align-items:start; }
@media (max-width:1100px) { .viewer-workspace { grid-template-columns:1fr; } }
__DEFLECTION_CSS__
__TRAJECTORY_CSS__
</style>
</head>
<body><main>
<h1>VIVE 偏转查看器</h1>
<p class="subtitle">只读取一个 VIVE wrist pose；不创建 sim、policy、HITL composition、authority、记录器或机器人执行链。</p>
<div class="toolbar">
  <button id="recenter">以当前姿态设为 Home</button>
  <button id="recalibrate">重新标定坐标系</button>
  <span id="calibration-status" class="subtle calibration-status">标定页未打开</span>
  <span id="source-status" class="subtle">等待 VIVE</span>
</div>
<p id="axis-warning" class="axis-warning" hidden></p>
<div class="viewer-workspace">
__DEFLECTION_HTML__
__TRAJECTORY_HTML__
</div>
</main>
<script nonce="__CSP_NONCE__">
"use strict";
const $ = (id) => document.getElementById(id);
const text = (value) => value === null || value === undefined || value === '' ? '—' : String(value);
const uiValue = text;
__DEFLECTION_JS__
__TRAJECTORY_JS__
const VIEWER_REFRESH_PERIOD_MS = 1000 / 60;
let refreshInFlight = false;
let refreshTimer = null;
let immediateRefreshRequested = false;
let pendingViewerState = null;
let viewerApplyScheduled = false;
let viewerResponseGeneration = 0;
let latestAppliedResponseGeneration = 0;
let lastViewerCoordinateProfile = null;
let viewerCoordinateProfileInitialized = false;

function applyLatestViewerState() {
  const applyStartedAt = performance.now();
  viewerApplyScheduled = false;
  const pending = pendingViewerState;
  pendingViewerState = null;
  if (!pending) return;
  if (pending.generation <= latestAppliedResponseGeneration) return;
  latestAppliedResponseGeneration = pending.generation;
  const value = pending.value;
  const source = value.source || {};
  const calibration = value.calibration || {};
  const coordinateProfile = source.coordinate_profile_path || null;
  if (viewerCoordinateProfileInitialized
      && coordinateProfile !== lastViewerCoordinateProfile
      && trajectoryGuidanceScene) {
    trajectoryGuidanceScene.clearIntegratedOffset();
    trajectoryGuidanceLastMetricText = null;
    renderTrajectoryGuidanceMetrics();
  }
  lastViewerCoordinateProfile = coordinateProfile;
  viewerCoordinateProfileInitialized = true;
  const sourceAge = source.age_sec === null || source.age_sec === undefined
    ? Number.NaN : Number(source.age_sec);
  const ageLabel = Number.isFinite(sourceAge)
    ? `${(sourceAge * 1000).toFixed(1)}ms` : '—';
  const rttLabel = Number.isFinite(pending.roundTripMs)
    ? `${pending.roundTripMs.toFixed(1)}ms` : '—';
  $('recenter').disabled = !Boolean(source.recenter_ready);
  $('recenter').title = source.recenter_ready
    ? '以当前新鲜有效 pose 更新 Viewer Home。'
    : text(source.message || '等待新鲜有效 VIVE pose。');
  $('recalibrate').disabled = calibration.available === false;
  const axisWarning = source.axis_semantics_warning;
  $('axis-warning').hidden = !axisWarning;
  $('axis-warning').textContent = axisWarning || '';
  $('calibration-status').textContent = calibration.apply_error
    ? `新标定应用失败：${text(calibration.apply_error)}`
    : calibration.last_applied_profile_path
      ? `新标定已保存并应用；Viewer Home 已重置`
      : calibration.active
        ? `标定页已打开 · ${text(calibration.source_role)}`
        : calibration.last_saved_profile_path
          ? `新标定已保存，等待应用`
          : '标定页未打开';
  const sampleToken = `${text(source.seq)}:${text(source.receiver_timestamp)}`;
  renderRelativeDeflection(value.deflection || {}, sampleToken);
  updateTrajectoryGuidance(value.deflection || {}, sampleToken);
  const optionalNumber = value => value === null || value === undefined
    ? Number.NaN : Number(value);
  const receiverTimestamp = optionalNumber(source.receiver_timestamp);
  const presentationAgeMs = Number.isFinite(receiverTimestamp)
    ? Math.max(0, Date.now() - receiverTimestamp * 1000) : Number.NaN;
  const transportAgeMs = optionalNumber(source.transport_datagram_age_sec) * 1000;
  const processDurationMs = optionalNumber(source.last_process_duration_sec) * 1000;
  const snapshotDurationMs = optionalNumber(value.snapshot_duration_sec) * 1000;
  const applyQueueMs = applyStartedAt - pending.responseReceivedAt;
  const renderMs = performance.now() - applyStartedAt;
  const ms = value => Number.isFinite(value) ? `${value.toFixed(1)}ms` : '—';
  $('source-status').textContent = [
    text(source.status),
    text(source.role),
    `seq ${text(source.seq)}`,
    `总呈现 ${ms(presentationAgeMs)}`,
    `源年龄 ${ageLabel}`,
    `socket ${ms(transportAgeMs)}`,
    `处理 ${ms(processDurationMs)}`,
    `snapshot ${ms(snapshotDurationMs)}`,
    `HTTP ${rttLabel}`,
    `UI等待 ${ms(applyQueueMs)}`,
    `绘制 ${ms(renderMs)}`,
  ].join(' · ');
}

function queueViewerState(value, roundTripMs) {
  viewerResponseGeneration += 1;
  pendingViewerState = {
    value,
    roundTripMs,
    responseReceivedAt: performance.now(),
    generation: viewerResponseGeneration,
  };
  if (!viewerApplyScheduled) {
    viewerApplyScheduled = true;
    window.requestAnimationFrame(applyLatestViewerState);
  }
}

function scheduleRefresh(delayMs = VIEWER_REFRESH_PERIOD_MS) {
  if (refreshInFlight) {
    immediateRefreshRequested = true;
    return;
  }
  if (refreshTimer !== null) window.clearTimeout(refreshTimer);
  refreshTimer = window.setTimeout(refresh, Math.max(0, delayMs));
}

async function refresh() {
  if (refreshInFlight) {
    immediateRefreshRequested = true;
    return;
  }
  refreshInFlight = true;
  refreshTimer = null;
  const startedAt = performance.now();
  try {
    const response = await fetch('/api/state', {cache:'no-store'});
    if (!response.ok) throw new Error(response.statusText);
    const value = await response.json();
    queueViewerState(value, performance.now() - startedAt);
  } catch (error) {
    $('source-status').textContent = `GUI offline: ${error}`;
  } finally {
    refreshInFlight = false;
    const elapsedMs = performance.now() - startedAt;
    const requestedImmediately = immediateRefreshRequested;
    immediateRefreshRequested = false;
    scheduleRefresh(requestedImmediately
      ? 0 : Math.max(0, VIEWER_REFRESH_PERIOD_MS - elapsedMs));
  }
}
$('recenter').addEventListener('click', async () => {
  const response = await fetch('/api/recenter', {method:'POST'});
  const value = await response.json();
  if (!response.ok) window.alert(value.error || 'recenter failed');
  if (response.ok && trajectoryGuidanceScene) {
    trajectoryGuidanceScene.clearIntegratedOffset();
    trajectoryGuidanceLastMetricText = null;
    renderTrajectoryGuidanceMetrics();
  }
  scheduleRefresh(0);
});
$('recalibrate').addEventListener('click', async () => {
  const popup = window.open('about:blank', 'vive-coordinate-calibration');
  try {
    const response = await fetch('/api/calibration/open', {method:'POST'});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'calibration launch failed');
    if (popup) popup.location.replace(value.url);
    else window.alert(`浏览器阻止了弹窗，请打开：${value.url}`);
    scheduleRefresh(0);
  } catch (error) {
    if (popup) popup.close();
    window.alert(String(error));
  }
});
scheduleRefresh(0);
</script></body></html>"""

_HTML = _HTML.replace("__DEFLECTION_CSS__", RELATIVE_DEFLECTION_VIEW_CSS)
_HTML = _HTML.replace("__TRAJECTORY_CSS__", TRAJECTORY_GUIDANCE_VIEW_CSS)
_HTML = _HTML.replace("__DEFLECTION_HTML__", RELATIVE_DEFLECTION_VIEW_HTML)
_HTML = _HTML.replace("__TRAJECTORY_HTML__", TRAJECTORY_GUIDANCE_VIEW_HTML)
_HTML = _HTML.replace("__DEFLECTION_JS__", RELATIVE_DEFLECTION_VIEW_JS)
_HTML = _HTML.replace("__TRAJECTORY_JS__", TRAJECTORY_GUIDANCE_VIEW_JS)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: Any,
        handler: Any,
        *,
        controller: Any,
        calibration_launcher: Any | None = None,
    ) -> None:
        self.controller = controller
        self.calibration_launcher = calibration_launcher
        super().__init__(address, handler)

    @staticmethod
    def _configure_low_latency_request_socket(request: socket.socket) -> None:
        request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def get_request(self) -> tuple[socket.socket, Any]:
        request, client_address = super().get_request()
        # This GUI sends small JSON responses over a persistent loopback
        # HTTP/1.1 connection.  Without TCP_NODELAY, Linux/Chromium can fall
        # into a roughly 40 ms Nagle/delayed-ACK cadence, reducing the nominal
        # 60 Hz viewer to about 13-20 Hz.  This socket is presentation-only;
        # disabling Nagle does not alter VIVE source or control semantics.
        self._configure_low_latency_request_socket(request)
        return request, client_address


class _Handler(BaseHTTPRequestHandler):
    server_version = "DexTeleopViveDeflection/1"
    protocol_version = "HTTP/1.1"

    @property
    def controller(self) -> ViveDeflectionController:
        return self.server.controller  # type: ignore[attr-defined, no-any-return]

    @property
    def calibration_launcher(self) -> Any | None:
        return self.server.calibration_launcher  # type: ignore[attr-defined, no-any-return]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        *,
        nonce: str | None = None,
    ) -> None:
        script_source = "'self'" if nonce is None else f"'self' 'nonce-{nonce}'"
        style_source = "'self'" if nonce is None else f"'self' 'nonce-{nonce}'"
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            f"script-src {script_source}; style-src {style_source}; "
            "connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: HTTPStatus, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, payload, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            nonce = secrets.token_urlsafe(18)
            page = _HTML.replace("__CSP_NONCE__", nonce).encode("utf-8")
            self._send(HTTPStatus.OK, page, "text/html; charset=utf-8", nonce=nonce)
        elif path == "/api/state":
            state = self.controller.snapshot()
            launcher = self.calibration_launcher
            state["calibration"] = (
                {"available": False, "active": False}
                if launcher is None
                else launcher.snapshot()
            )
            self._json(HTTPStatus.OK, state)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/calibration/open":
            launcher = self.calibration_launcher
            if launcher is None:
                self._json(
                    HTTPStatus.NOT_IMPLEMENTED,
                    {"error": "calibration launcher is unavailable"},
                )
                return
            try:
                state = launcher.open()
                self._json(HTTPStatus.OK, state)
            except Exception as error:
                self._json(
                    HTTPStatus.CONFLICT,
                    {"error": f"{type(error).__name__}:{error}"},
                )
            return
        if path != "/api/recenter":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            self._json(HTTPStatus.OK, self.controller.recenter())
        except Exception as error:
            self._json(
                HTTPStatus.CONFLICT,
                {"error": f"{type(error).__name__}:{error}"},
            )


class ViveDeflectionGui:
    def __init__(
        self,
        controller: ViveDeflectionController,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        calibration_launcher: Any | None = None,
    ) -> None:
        self.controller = controller
        self.host = str(host).strip().lower()
        if self.host not in _LOOPBACK_HOSTS:
            raise ValueError("VIVE deflection GUI must use 127.0.0.1 or localhost")
        self.port = int(port)
        self.open_browser = bool(open_browser)
        self.calibration_launcher = calibration_launcher
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        port = self.port if self._server is None else self._server.server_port
        return f"http://{self.host}:{port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _Server(
            (self.host, self.port),
            _Handler,
            controller=self.controller,
            calibration_launcher=self.calibration_launcher,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="vive_deflection_gui",
        )
        self._thread.start()
        if self.open_browser:
            webbrowser.open(self.url)

    def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)


__all__ = ["ViveDeflectionGui"]
