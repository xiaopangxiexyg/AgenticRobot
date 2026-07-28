"""Read-only status surface for the Control Station Sim-HIL profile.

The first Sim-HIL GUI is intentionally observational.  It exposes readiness,
source freshness, composition, and execution receipts, but it has no control
POST endpoint and does not own a queue, authority lease, safety transition, or
dispatch path.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit


StatusProvider = Callable[[], dict[str, Any]]
SIM_HIL_STATUS_SCHEMA_V1 = "dexteleop.control_station.sim_hil_status.v1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>DexTeleop · Sim-HIL</title>
  <style>
    :root {
      --bg: #0a0f1c; --panel: #111827; --panel-2: #172033; --line: #26344d;
      --text: #e5edf8; --muted: #91a0b8; --blue: #73b7ff; --green: #69d39a;
      --amber: #f3bd62; --red: #ff7e8a; --shadow: 0 14px 36px rgba(0,0,0,.22);
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text); background: radial-gradient(circle at 15% 0%, #182744, var(--bg) 40%); font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
    main { max-width: 1440px; margin: 0 auto; padding: 28px 24px 42px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 22px; }
    .kicker { color: var(--blue); font-size: 11px; letter-spacing: .16em; font-weight: 700; }
    h1 { margin: 5px 0 5px; font-size: clamp(26px, 4vw, 42px); letter-spacing: -.03em; }
    h2 { margin: 0; font-size: 15px; }
    p { margin: 0; color: var(--muted); }
    .badge { min-width: 130px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 14px; background: rgba(17,24,39,.82); box-shadow: var(--shadow); text-align: center; font-weight: 800; letter-spacing: .08em; }
    .badge.ready { color: var(--green); border-color: rgba(105,211,154,.45); }
    .badge.hold { color: var(--amber); border-color: rgba(243,189,98,.45); }
    .badge.error { color: var(--red); border-color: rgba(255,126,138,.5); }
    .error { display: none; margin: 0 0 18px; padding: 13px 15px; color: #ffd6da; border: 1px solid rgba(255,126,138,.5); border-radius: 12px; background: rgba(127,29,45,.3); }
    .error.visible { display: block; }
    .grid { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 14px; }
    .card { grid-column: span 4; min-width: 0; padding: 17px; border: 1px solid var(--line); border-radius: 16px; background: linear-gradient(145deg, rgba(23,32,51,.96), rgba(17,24,39,.94)); box-shadow: var(--shadow); }
    .card.wide { grid-column: span 8; }
    .card.full { grid-column: 1 / -1; }
    .card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 13px; }
    .label { color: var(--muted); font-size: 12px; }
    dl { display: grid; grid-template-columns: minmax(120px, .8fr) minmax(0, 1.4fr); gap: 9px 12px; margin: 0; }
    dt { color: var(--muted); overflow-wrap: anywhere; }
    dd { margin: 0; color: var(--text); overflow-wrap: anywhere; text-align: right; }
    dd.mono, pre, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .checks { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; margin: 0; padding: 0; list-style: none; }
    .check { display: flex; align-items: flex-start; gap: 9px; padding: 9px 10px; border-radius: 10px; background: rgba(10,15,28,.5); }
    .dot { flex: 0 0 auto; width: 9px; height: 9px; margin-top: 5px; border-radius: 50%; background: var(--red); }
    .dot.ok { background: var(--green); box-shadow: 0 0 10px rgba(105,211,154,.6); }
    .check-name { font-weight: 650; }
    .check-detail { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .source-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 13px; }
    .source { padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(10,15,28,.38); }
    .source h3 { margin: 0 0 9px; font-size: 13px; color: var(--blue); }
    .source dl { grid-template-columns: 1fr 1fr; gap: 6px 8px; font-size: 12px; }
    .source dd { text-align: right; }
    .subtle { color: var(--muted); }
    details { margin-top: 14px; }
    summary { cursor: pointer; color: var(--blue); }
    pre { max-height: 420px; margin: 12px 0 0; padding: 14px; overflow: auto; border: 1px solid var(--line); border-radius: 12px; color: #c9d7ea; background: #080c15; white-space: pre-wrap; }
    @media (max-width: 980px) { .card, .card.wide { grid-column: span 6; } }
    @media (max-width: 680px) { main { padding: 20px 14px 32px; } header { display: block; } .badge { margin-top: 15px; } .card, .card.wide { grid-column: 1 / -1; } .checks, .source-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="kicker">CONTROL STATION / SIM-HIL</div>
        <h1>MuJoCo 混合控制状态</h1>
        <p>只读观察面：PPO 与 VIVE/Q6 仅作为 contributor，MuJoCo composer 是唯一执行 owner。</p>
      </div>
      <div id="badge" class="badge hold">LOADING</div>
    </header>
    <div id="error" class="error"></div>
    <section class="grid">
      <article class="card">
        <div class="card-head"><h2>Profile</h2><span id="sampled" class="label">—</span></div>
        <dl id="profile-rows"></dl>
      </article>
      <article class="card wide">
        <div class="card-head"><h2>Readiness gate</h2><span id="readiness-label" class="label">—</span></div>
        <ul id="checks" class="checks"></ul>
      </article>
      <article class="card full">
        <div class="card-head"><h2>Contributors</h2><span class="label">no raw action values</span></div>
        <div class="source-grid">
          <div class="source"><h3>Temporary PPO</h3><dl id="policy-rows"></dl></div>
          <div class="source"><h3>Passive VIVE + Q6</h3><dl id="human-rows"></dl></div>
        </div>
      </article>
      <article class="card wide">
        <div class="card-head"><h2>Composition</h2><span id="composition-label" class="label">—</span></div>
        <dl id="composition-rows"></dl>
      </article>
      <article class="card">
        <div class="card-head"><h2>Execution</h2><span id="gate-label" class="label">—</span></div>
        <dl id="execution-rows"></dl>
      </article>
      <article class="card full">
        <details>
          <summary>查看完整 status JSON（仅协议元数据与 receipt 摘要）</summary>
          <pre id="raw-status">—</pre>
        </details>
      </article>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const text = (value) => {
      if (value === null || value === undefined || value === '') return '—';
      if (typeof value === 'object') return JSON.stringify(value);
      return String(value);
    };
    const short = (value) => {
      const rendered = text(value);
      return rendered.length > 96 ? rendered.slice(0, 92) + '…' : rendered;
    };
    function rows(target, entries) {
      target.replaceChildren();
      for (const [name, value] of entries) {
        const dt = document.createElement('dt');
        const dd = document.createElement('dd');
        dt.textContent = name;
        dd.textContent = short(value);
        if (typeof value === 'string' && (value.includes('sha') || value.length > 48)) dd.className = 'mono';
        target.append(dt, dd);
      }
    }
    function renderChecks(readiness) {
      const target = $('checks');
      target.replaceChildren();
      const details = readiness.detail_by_check || {};
      const names = Object.keys(details);
      for (const name of names) {
        const item = document.createElement('li');
        item.className = 'check';
        const dot = document.createElement('span');
        const ok = !(readiness.failed_checks || []).includes(name);
        dot.className = ok ? 'dot ok' : 'dot';
        const body = document.createElement('div');
        const title = document.createElement('div');
        const detail = document.createElement('div');
        title.className = 'check-name';
        detail.className = 'check-detail';
        title.textContent = name;
        detail.textContent = details[name] || (ok ? 'ready' : 'failed');
        body.append(title, detail);
        item.append(dot, body);
        target.append(item);
      }
      $('readiness-label').textContent = readiness.ready ? 'ALL READY' : `${(readiness.failed_checks || []).length} CHECK(S) FAILED`;
    }
    function render(value) {
      $('error').className = 'error';
      const readiness = value.readiness || {};
      const gate = value.execution_gate || {};
      const ready = Boolean(value.started && readiness.ready && gate.status === 'ready');
      const badge = $('badge');
      badge.textContent = value.started ? (ready ? 'READY' : 'HOLD') : 'STOPPED';
      badge.className = `badge ${ready ? 'ready' : 'hold'}`;
      $('sampled').textContent = `t=${text(value.sampled_at)}`;
      const binding = value.calibration || {};
      const contracts = value.contracts || {};
      const targetContract = contracts.target || {};
      rows($('profile-rows'), [
        ['lifecycle', value.lifecycle], ['profile', value.profile_id],
        ['target', value.execution_target], ['real arm motion', value.real_arm_motion],
        ['owner', value.single_execution_owner], ['status schema', value.schema],
        ['Home profile', binding.home_profile_path], ['binding', binding.mode],
        ['VIVE coordinate', binding.vive_coordinate_profile_path],
        ['target handoff', targetContract.handoff_enabled],
        ['target activation', targetContract.activation_mode],
        ['target capability', targetContract.component_target_capability_hash],
      ]);
      renderChecks(readiness);
      const policy = value.policy || {};
      const policyEmission = policy.last_emission || {};
      rows($('policy-rows'), [
        ['state', policy.started ? (policy.ready ? 'ready' : 'fault') : 'stopped'],
        ['checkpoint', policy.checkpoint], ['queue', policy.queue_length],
        ['last emission', policyEmission.emission_id],
        ['emission hash', policyEmission.semantic_content_hash],
        ['payload kind', policyEmission.action_payload_kind],
        ['representation trace', policyEmission.representation_trace_id],
        ['canonicalization', policyEmission.canonicalization_receipt_hash],
        ['TTL age', policy.last_emission_age_sec], ['last error', policy.last_error],
      ]);
      const human = value.human || {};
      const humanEmission = human.last_emission || {};
      const bindingStatus = human.binding || {};
      const safety = human.session_safety || {};
      rows($('human-rows'), [
        ['session', human.session_state], ['source started', human.source_started],
        ['last emission', humanEmission.emission_id],
        ['emission hash', humanEmission.semantic_content_hash],
        ['age', human.last_emission_age_sec], ['rig absent', bindingStatus.rig_absent],
        ['authoritative', bindingStatus.authoritative],
        ['hold', safety.motion_hold_mode || 'none'],
        ['home re-calibration', safety.home_recalibration_required],
        ['VIVE port', (value.source_configuration || {}).vive_port],
        ['Q6 ports', `${(value.source_configuration || {}).left_q6_port || '—'} / ${(value.source_configuration || {}).right_q6_port || '—'}`],
        ['channels', Object.keys(human.channels || {}).join(', ')],
      ]);
      const composition = value.composition || {};
      const activation = composition.activation || {};
      const last = composition.last_cycle || {};
      const step = last.composition_step || {};
      const disposition = step.disposition || {};
      const proposal = step.proposal || {};
      rows($('composition-rows'), [
        ['started', composition.started], ['activation', activation.activation_id],
        ['activation hash', activation.semantic_content_hash],
        ['cycle', last.cycle_index], ['cycle hash', last.semantic_content_hash],
        ['support until', (step.composition_evaluation || {}).composition_supported_until],
        ['disposition', disposition.downstream_request],
        ['proposal', proposal.proposal_id], ['proposal until', proposal.effective_until],
        ['human input', (last.human_submission || {}).accepted],
        ['policy input', (last.policy_submission || {}).accepted],
        ['blocked', step.blocked_reason || step.error],
      ]);
      const execution = last.execution || {};
      rows($('execution-rows'), [
        ['gate', gate.status], ['gate reason', gate.reason],
        ['gate epoch', gate.gate_epoch], ['cycle execution', execution.status],
        ['command', execution.command_id], ['dispatch receipt', execution.dispatch_receipt_id],
        ['target request', execution.target_execution_request_id],
        ['target submission', execution.target_submission_receipt_id],
        ['target completion', execution.target_completion_receipt_id],
        ['component target', execution.component_execution_target_id],
        ['target admission', execution.component_target_admission_receipt_id],
        ['target activation', execution.component_target_activation_id],
        ['target support until', execution.target_supported_until],
        ['last error', value.last_error],
      ]);
      $('composition-label').textContent = last.cycle_index === undefined ? 'WAITING' : `CYCLE ${last.cycle_index}`;
      $('gate-label').textContent = gate.status || 'not started';
      $('raw-status').textContent = JSON.stringify(value, null, 2);
    }
    function renderError(error) {
      $('badge').textContent = 'STATUS ERROR';
      $('badge').className = 'badge error';
      $('error').textContent = String(error);
      $('error').className = 'error visible';
    }
    async function refresh() {
      try {
        const response = await fetch('/api/status', {cache: 'no-store'});
        const payload = await response.text();
        let value;
        try { value = JSON.parse(payload); } catch (error) { throw new Error(`invalid status JSON: ${error}`); }
        if (!response.ok) throw new Error(value.status_error || `status HTTP ${response.status}`);
        if (value.status_error) throw new Error(value.status_error);
        render(value);
      } catch (error) {
        renderError(error);
      }
    }
    refresh();
    setInterval(refresh, 800);
  </script>
</body>
</html>
"""


class SimHILStatusServer:
    """Serve a read-only status page around an application status callable."""

    def __init__(
        self,
        status_provider: StatusProvider,
        *,
        host: str = "127.0.0.1",
        port: int = 8787,
    ) -> None:
        if not callable(status_provider):
            raise TypeError("status_provider must be callable.")
        self.status_provider = status_provider
        self.host = str(host).strip().lower()
        self.port = int(port)
        if self.host not in _LOOPBACK_HOSTS:
            raise ValueError(
                "Sim-HIL status GUI must use a loopback host: 127.0.0.1 or localhost."
            )
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be in [0, 65535].")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def started(self) -> bool:
        return self._server is not None

    @property
    def bound_address(self) -> tuple[str, int] | None:
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            address = self.bound_address
            assert address is not None
            return address

        provider = self.status_provider

        class Handler(BaseHTTPRequestHandler):
            server_version = "DexTeleopSimHILStatus/1"

            def _write(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'unsafe-inline'; "
                    "style-src 'unsafe-inline'; connect-src 'self'; "
                    "base-uri 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
                path = urlsplit(self.path).path
                if path == "/":
                    self._write(200, "text/html; charset=utf-8", _HTML.encode("utf-8"))
                    return
                if path == "/api/status":
                    try:
                        value = provider()
                        if not isinstance(value, dict):
                            raise TypeError("status provider must return a dict.")
                        payload = json.dumps(
                            value,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        ).encode("utf-8")
                        self._write(200, "application/json; charset=utf-8", payload)
                    except Exception as error:  # noqa: BLE001 - status must stay observable.
                        payload = json.dumps(
                            {
                                "schema": SIM_HIL_STATUS_SCHEMA_V1,
                                "status_error": f"{type(error).__name__}:{error}",
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode("utf-8")
                        self._write(500, "application/json; charset=utf-8", payload)
                    return
                self._write(404, "text/plain; charset=utf-8", b"not found\n")

            def do_POST(self) -> None:  # noqa: N802 - explicitly no control API.
                self.send_response(405)
                self.send_header("Allow", "GET")
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(b"read-only status server\n")))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(b"read-only status server\n")

            def log_message(self, _format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="control_station_sim_hil_status",
            daemon=True,
        )
        self._thread.start()
        address = self.bound_address
        assert address is not None
        return address

    def close(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=1.0)


__all__ = ["SimHILStatusServer"]
