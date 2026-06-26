from __future__ import annotations

import json
import queue
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VIVE -> RM65 Calibration</title>
<style>
:root {
  color-scheme: light;
  --bg: #0b1220;
  --panel: #111a2d;
  --panel-2: #16233b;
  --text: #e8eefc;
  --muted: #90a4c7;
  --accent: #66d9ef;
  --accent-2: #ffb86b;
  --good: #5fe1a5;
  --warn: #ffd166;
  --danger: #ff6b6b;
  --border: rgba(255, 255, 255, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background:
    radial-gradient(circle at top left, rgba(102, 217, 239, 0.14), transparent 28%),
    radial-gradient(circle at 85% 20%, rgba(255, 184, 107, 0.14), transparent 26%),
    linear-gradient(180deg, #0b1220 0%, #0e1526 48%, #09101d 100%);
  min-height: 100vh;
}
header {
  padding: 22px 24px 14px;
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(8px);
}
h1 {
  margin: 0;
  font-size: 24px;
  letter-spacing: 0.02em;
}
.subtitle {
  margin-top: 6px;
  color: var(--muted);
  font-size: 13px;
}
main {
  display: grid;
  gap: 16px;
  grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.9fr);
  padding: 16px 24px 24px;
}
.panel {
  background: linear-gradient(180deg, rgba(22, 35, 59, 0.96), rgba(17, 26, 45, 0.96));
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 24px 50px rgba(0, 0, 0, 0.24);
  overflow: hidden;
}
.panel header {
  border: 0;
  padding: 16px 18px 0;
}
.panel h2 {
  margin: 0;
  font-size: 15px;
}
.panel-body {
  padding: 16px 18px 18px;
}
.status-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.metric {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px;
}
.label { color: var(--muted); font-size: 12px; }
.value { margin-top: 6px; font-size: 18px; font-weight: 700; }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.5;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid var(--border);
  color: var(--muted);
  margin-right: 8px;
  margin-bottom: 8px;
}
.pill strong { color: var(--text); font-weight: 600; }
.stage {
  font-size: 20px;
  font-weight: 800;
  letter-spacing: 0.02em;
  margin: 8px 0 4px;
}
.message-box {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px;
  min-height: 180px;
  max-height: 380px;
  overflow: auto;
}
.message {
  padding: 8px 10px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.04);
  margin-bottom: 8px;
}
.message .tag { color: var(--accent); font-weight: 700; margin-right: 8px; }
.button-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
button {
  appearance: none;
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(102, 217, 239, 0.16), rgba(102, 217, 239, 0.08));
  color: var(--text);
  border-radius: 12px;
  padding: 12px 10px;
  cursor: pointer;
  font: inherit;
  transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
}
button:hover { transform: translateY(-1px); border-color: rgba(102, 217, 239, 0.45); }
button.secondary {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.03));
}
button.warn {
  background: linear-gradient(180deg, rgba(255, 184, 107, 0.20), rgba(255, 184, 107, 0.08));
}
button.danger {
  background: linear-gradient(180deg, rgba(255, 107, 107, 0.20), rgba(255, 107, 107, 0.08));
}
.section { margin-top: 14px; }
.hint {
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
@media (max-width: 1100px) {
  main { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>VIVE -> RM65 Calibration</h1>
  <div class="subtitle">同一套标定逻辑，网页只负责发出与终端一致的控制事件。</div>
</header>
<main>
  <section class="panel">
    <header><h2>Current State</h2></header>
    <div class="panel-body">
      <div id="stage" class="stage">Starting...</div>
      <div id="pills"></div>
      <div class="status-grid section">
        <div class="metric"><div class="label">Selected Side</div><div id="selectedSide" class="value">-</div></div>
        <div class="metric"><div class="label">Frame Viz Mode</div><div id="frameVizMode" class="value">-</div></div>
        <div class="metric"><div class="label">Virtual Desk Yaw</div><div id="virtualDeskYaw" class="value">-</div></div>
        <div class="metric"><div class="label">Renderer</div><div id="rendererState" class="value">-</div></div>
      </div>
      <div class="section">
        <div class="label">Latest Messages</div>
        <div id="messages" class="message-box mono"></div>
      </div>
    </div>
  </section>
  <section class="panel">
    <header><h2>Controls</h2></header>
    <div class="panel-body">
      <div class="button-grid">
        <button class="danger" data-event="q">Quit</button>
        <button class="warn" data-event="r">Reset</button>
        <button data-event="c">Capture</button>
        <button class="secondary" data-event="n">Next</button>
        <button class="secondary" data-event="s">Save</button>
        <button class="secondary" data-event="h">Home</button>
        <button data-event="1">Left</button>
        <button data-event="2">Right</button>
        <button data-event="v">Cycle View</button>
        <button data-event="j">Yaw -</button>
        <button data-event="l">Yaw +</button>
        <button data-event="i">Pitch +</button>
        <button data-event="k">Pitch -</button>
        <button data-event="u">Roll -</button>
        <button data-event="o">Roll +</button>
        <button data-event="x">Reset Tune</button>
        <button data-event="f">Flip XY</button>
      </div>
      <div class="hint section">
        按钮对应原始按键事件，内部仍然走同一套状态机。<br>
        如果需要回到终端操作，可以用 <strong>--ui-mode terminal</strong>。
      </div>
    </div>
  </section>
</main>
<script>
const buttons = document.querySelectorAll("button[data-event]");
const stageEl = document.getElementById("stage");
const pillsEl = document.getElementById("pills");
const messagesEl = document.getElementById("messages");
const selectedSideEl = document.getElementById("selectedSide");
const frameVizModeEl = document.getElementById("frameVizMode");
const virtualDeskYawEl = document.getElementById("virtualDeskYaw");
const rendererStateEl = document.getElementById("rendererState");

async function sendEvent(eventName) {
  await fetch("/event", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({event: eventName}),
  });
}

buttons.forEach((button) => {
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await sendEvent(button.dataset.event);
    } finally {
      button.disabled = false;
    }
  });
});

function formatPills(state) {
  const items = [];
  if (state.ui_mode) items.push(["UI", state.ui_mode]);
  if (state.host) items.push(["VIVE", `${state.host}:${state.port}`]);
  if (state.load_profile) items.push(["Profile", state.load_profile]);
  if (state.show_tracker_frame_viz) items.push(["Viz", "enabled"]);
  return items.map(([label, value]) => `<span class="pill"><strong>${label}</strong>${value}</span>`).join("");
}

function renderMessages(messages) {
  if (!messages || messages.length === 0) {
    messagesEl.innerHTML = '<div class="message">waiting for updates...</div>';
    return;
  }
  messagesEl.innerHTML = messages.map((message) => {
    const tag = message.tag || "LOG";
    const text = message.message || "";
    return `<div class="message"><span class="tag">[${tag}]</span>${text}</div>`;
  }).join("");
}

async function refresh() {
  const response = await fetch("/state.json?ts=" + Date.now(), {cache: "no-store"});
  const state = await response.json();
  stageEl.textContent = state.stage || "Starting...";
  pillsEl.innerHTML = formatPills(state);
  selectedSideEl.textContent = state.selected_side || "-";
  frameVizModeEl.textContent = state.frame_viz_mode || "-";
  virtualDeskYawEl.textContent = state.virtual_desk_post_yaw_deg !== undefined ? String(state.virtual_desk_post_yaw_deg) + " deg" : "-";
  rendererStateEl.textContent = state.renderer_ready ? "ready" : "deferred";
  renderMessages(state.messages || []);
}

refresh();
setInterval(refresh, 300);
</script>
</body>
</html>
"""


@dataclass
class CalibrationWebUiState:
    stage: str = "Starting..."
    selected_side: str = "left"
    frame_viz_mode: str = "calibrated"
    virtual_desk_post_yaw_deg: float = 180.0
    renderer_ready: bool = False
    ui_mode: str = "web"
    host: str = ""
    port: int = 0
    load_profile: str = ""
    show_tracker_frame_viz: bool = False
    messages: list[dict[str, str]] = field(default_factory=list)


class CalibrationWebUi:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._open_browser = bool(open_browser)
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._events: queue.Queue[str] = queue.Queue()
        self._state_lock = threading.Lock()
        self._state = CalibrationWebUiState()

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self._host}:{self._port}/"
        address = self._server.server_address
        host = address[0]
        port = address[1]
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}/"

    def start(self) -> None:
        if self._server is not None:
            return

        ui = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/state.json"):
                    payload = json.dumps(ui.snapshot()).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if self.path in {"/", "/index.html"}:
                    payload = _HTML_PAGE.encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", payload)
                    return
                self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/event":
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    event = str(payload.get("event", "")).strip()
                except Exception:
                    self._send(400, "text/plain; charset=utf-8", b"invalid json")
                    return
                if not event:
                    self._send(400, "text/plain; charset=utf-8", b"missing event")
                    return
                ui.push_event(event)
                self._send(204, "text/plain; charset=utf-8", b"")

            def log_message(self, fmt: str, *args: Any) -> None:
                del fmt, args

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        if self._open_browser:
            try:
                webbrowser.open(self.url, new=1, autoraise=True)
            except Exception:
                pass

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._server_thread = None

    def push_event(self, event: str) -> None:
        self._events.put(str(event))

    def consume_events(self) -> list[str]:
        events: list[str] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def publish(self, **updates: Any) -> None:
        with self._state_lock:
            for key, value in updates.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)

    def append_message(self, tag: str, message: str) -> None:
        with self._state_lock:
            self._state.messages.append({"tag": str(tag), "message": str(message)})
            if len(self._state.messages) > 48:
                self._state.messages = self._state.messages[-48:]

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "stage": self._state.stage,
                "selected_side": self._state.selected_side,
                "frame_viz_mode": self._state.frame_viz_mode,
                "virtual_desk_post_yaw_deg": self._state.virtual_desk_post_yaw_deg,
                "renderer_ready": self._state.renderer_ready,
                "ui_mode": self._state.ui_mode,
                "host": self._state.host,
                "port": self._state.port,
                "load_profile": self._state.load_profile,
                "show_tracker_frame_viz": self._state.show_tracker_frame_viz,
                "messages": list(self._state.messages),
            }


class WebUiKeyInput:
    def __init__(self, ui: CalibrationWebUi) -> None:
        self.ui = ui
        self.enabled = True
        self.should_quit = False
        self._events: list[str] = []

    def start(self) -> None:
        self.ui.start()

    def close(self) -> None:
        return

    def poll(self) -> None:
        events = self.ui.consume_events()
        if not events:
            return
        self._events.extend(events)
        if "q" in events:
            self.should_quit = True

    def consume_events(self) -> list[str]:
        events = list(self._events)
        self._events.clear()
        return events
