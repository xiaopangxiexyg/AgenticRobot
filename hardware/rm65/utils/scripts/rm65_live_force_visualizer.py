from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65
from hardware.rm65.standard_paths import resolve_fit_path
from rm65_force_payload_calibrate import (
    GRAVITY_BASE,
    _cross,
    _mat_vec,
    _rotation_from_pose,
    _transpose,
)


FORCE_SOURCES = ("raw", "zero", "work", "tool")


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RM65 Live Corrected Force</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; color: #1f2933; background: #f7f8fa; }
header { padding: 14px 18px; background: #20262e; color: white; }
h1 { font-size: 18px; margin: 0; }
main { display: grid; grid-template-columns: minmax(420px, 1fr) 360px; gap: 16px; padding: 16px; }
.panel { background: white; border: 1px solid #d9dee5; padding: 14px; }
.panel h2 { font-size: 14px; margin: 0 0 10px; }
.values { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.metric { border: 1px solid #e1e5ea; padding: 8px; background: #fbfcfd; }
.metric .label { color: #52606d; font-size: 12px; }
.metric .value { font-size: 18px; font-weight: 600; margin-top: 3px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
#status { color: #dce5ef; font-size: 12px; margin-top: 4px; }
.ok { color: #1f7a4d; }
.warn { color: #b45309; }
.bad { color: #b91c1c; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
td, th { border-bottom: 1px solid #e5e7eb; padding: 5px; text-align: left; }
svg { width: 100%; max-height: 560px; background: #fafafa; border: 1px solid #e1e5ea; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
<h1>RM65 Live Corrected Force, Base Frame</h1>
<div id="status">connecting...</div>
</header>
<main>
<section class="panel">
<h2>3D Force Vectors</h2>
<svg id="viz" viewBox="0 0 720 560" role="img"></svg>
</section>
<section class="panel">
<h2>Live Values</h2>
<div class="values">
  <div class="metric"><div class="label">Corrected |F|</div><div class="value" id="corrNorm">--</div></div>
  <div class="metric"><div class="label">Measured |F|</div><div class="value" id="measNorm">--</div></div>
  <div class="metric"><div class="label">Source</div><div class="value" id="source">--</div></div>
  <div class="metric"><div class="label">Rate</div><div class="value" id="rate">--</div></div>
</div>
<h2 style="margin-top:14px">Corrected Force N</h2>
<div class="mono" id="corrVec">--</div>
<h2 style="margin-top:14px">Pose xyz/rpy</h2>
<div class="mono" id="pose">--</div>
<h2 style="margin-top:14px">Force Sources</h2>
<table id="sources"><thead><tr><th>source</th><th>|F| N</th><th>|T| Nm</th></tr></thead><tbody></tbody></table>
<h2 style="margin-top:14px">Warnings</h2>
<div class="mono" id="warnings">--</div>
</section>
</main>
<script>
const svg = document.getElementById("viz");
const statusEl = document.getElementById("status");
let lastSeq = -1;
let lastTime = performance.now();

function fmt(v, digits=3) {
  if (!Number.isFinite(v)) return "--";
  return v.toFixed(digits);
}
function fmtVec(v, digits=3) {
  if (!Array.isArray(v)) return "--";
  return "[" + v.map(x => fmt(x, digits)).join(", ") + "]";
}
function norm(v) {
  return Math.sqrt(v.reduce((a, x) => a + x * x, 0));
}
function project(p) {
  const scale = 125;
  const ox = 360;
  const oy = 300;
  return [ox + scale * (0.866 * p[0] - 0.866 * p[1]),
          oy + scale * (0.5 * p[0] + 0.5 * p[1] - p[2])];
}
function arrow(start, vector, color, label, forceScale) {
  const end = [start[0] + vector[0] * forceScale,
               start[1] + vector[1] * forceScale,
               start[2] + vector[2] * forceScale];
  const [x1, y1] = project(start);
  const [x2, y2] = project(end);
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 11;
  const lx = x2 - head * Math.cos(angle - 0.45);
  const ly = y2 - head * Math.sin(angle - 0.45);
  const rx = x2 - head * Math.cos(angle + 0.45);
  const ry = y2 - head * Math.sin(angle + 0.45);
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="3"/>`
    + `<polygon points="${x2},${y2} ${lx},${ly} ${rx},${ry}" fill="${color}"/>`
    + `<text x="${x2 + 8}" y="${y2 - 8}" fill="${color}" font-size="13">${label}</text>`;
}
function draw(data) {
  const pos = data.pose ? data.pose.slice(0, 3) : [0, 0, 0];
  const corr = data.corrected_force_base_n || [0, 0, 0];
  const meas = data.measured_force_base_n || [0, 0, 0];
  const forceScale = 0.045;
  let html = '<rect x="0" y="0" width="720" height="560" fill="#fafafa"/>';
  html += arrow([0,0,0], [0.45,0,0], "#d33", "Base X", 1);
  html += arrow([0,0,0], [0,0.45,0], "#2b8a3e", "Base Y", 1);
  html += arrow([0,0,0], [0,0,0.45], "#2563eb", "Base Z", 1);
  html += '<circle cx="' + project(pos)[0] + '" cy="' + project(pos)[1] + '" r="5" fill="#111"/>';
  html += arrow(pos, meas, "#c2410c", "measured", forceScale);
  html += arrow(pos, corr, "#047857", "corrected", forceScale);
  svg.innerHTML = html;
}
async function tick() {
  try {
    const res = await fetch("/state.json?ts=" + Date.now(), {cache: "no-store"});
    const data = await res.json();
    const now = performance.now();
    const hz = data.sequence !== lastSeq ? 1000 / Math.max(1, now - lastTime) : 0;
    if (data.sequence !== lastSeq) {
      lastSeq = data.sequence;
      lastTime = now;
    }
    const age = Date.now() / 1000 - data.timestamp;
    statusEl.textContent = `${data.arm} ${data.ip} seq=${data.sequence} age=${fmt(age, 2)}s`;
    statusEl.className = age < 1 ? "ok" : "warn";
    const cn = data.corrected_force_norm_n;
    document.getElementById("corrNorm").textContent = fmt(cn, 3) + " N";
    document.getElementById("corrNorm").className = "value " + (cn > data.warn_force_n ? "bad" : "ok");
    document.getElementById("measNorm").textContent = fmt(data.measured_force_norm_n, 3) + " N";
    document.getElementById("source").textContent = data.fit_source || "--";
    document.getElementById("rate").textContent = hz ? fmt(hz, 1) + " Hz" : "--";
    document.getElementById("corrVec").textContent = fmtVec(data.corrected_force_base_n, 4);
    document.getElementById("pose").textContent = fmtVec(data.pose, 4);
    document.getElementById("warnings").textContent = (data.warnings || []).join("\\n") || "--";
    const tbody = document.querySelector("#sources tbody");
    tbody.innerHTML = "";
    for (const [name, item] of Object.entries(data.force_sources || {})) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${name}</td><td>${fmt(item.force_norm_n, 3)}</td><td>${fmt(item.torque_norm_nm, 4)}</td>`;
      tbody.appendChild(tr);
    }
    draw(data);
  } catch (err) {
    statusEl.textContent = "disconnected: " + err;
    statusEl.className = "bad";
  }
}
setInterval(tick, 200);
tick();
</script>
</body>
</html>
"""


def _default_fit_path(side: str, tool_label: str | None) -> Path:
    return resolve_fit_path(side, tool_label, require_existing=True)


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _force_norms(wrench: list[float]) -> dict[str, float]:
    return {
        "force_norm_n": _norm(wrench[:3]),
        "torque_norm_nm": _norm(wrench[3:]),
    }


def _read_once(robot: Arm, side: str) -> dict[str, object]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    force = {
        "raw": [float(value) for value in raw],
        "zero": [float(value) for value in zero],
        "work": [float(value) for value in work],
        "tool": [float(value) for value in tool],
    }
    return {
        "joint_deg": [float(value) for value in joint_deg],
        "pose": [float(value) for value in pose],
        "force": force,
    }


def _load_fit(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise RuntimeError(f"{path} does not contain a fit object")
    if fit.get("mode") not in ("tool_gravity_model", "base_constant_offset"):
        raise RuntimeError(f"unsupported fit mode: {fit.get('mode')}")
    return payload


def _apply_correction(reading: dict[str, object], fit_payload: dict[str, object]) -> dict[str, object]:
    fit = fit_payload["fit"]
    mode = fit["mode"]
    source = fit["source"]
    pose = reading["pose"]
    force_sources = {
        name: {
            "vector": wrench,
            **_force_norms(wrench),
        }
        for name, wrench in reading["force"].items()
    }
    warnings = list(fit.get("warnings", []))

    if mode == "base_constant_offset":
        measured_base = reading["force"][source][:3]
        offset = [float(value) for value in fit["force_offset_base_n"]]
        corrected_base = [measured_base[i] - offset[i] for i in range(3)]
        predicted_frame = offset
        measured_frame = measured_base
    else:
        order = fit["euler_order"]
        rotation_base_tool = _rotation_from_pose(pose, order)
        rotation_tool_base = _transpose(rotation_base_tool)
        gravity_tool = _mat_vec(rotation_tool_base, GRAVITY_BASE)
        measured_frame = reading["force"][source][:3]
        force_bias = [float(value) for value in fit["force_bias_tool_n"]]
        signed_weight = float(fit["signed_weight_n"])
        predicted_frame = [
            force_bias[i] + signed_weight * gravity_tool[i] for i in range(3)
        ]
        corrected_frame = [
            measured_frame[i] - predicted_frame[i] for i in range(3)
        ]
        corrected_base = _mat_vec(rotation_base_tool, corrected_frame)
        measured_base = _mat_vec(rotation_base_tool, measured_frame)

        torque = reading["force"][source][3:]
        torque_bias = [float(value) for value in fit.get("torque_bias_tool_nm", [0, 0, 0])]
        weighted_com = [float(value) for value in fit.get("weighted_com_n_m", [0, 0, 0])]
        predicted_torque = [
            torque_bias[i] + _cross(weighted_com, gravity_tool)[i] for i in range(3)
        ]
        corrected_torque = [
            torque[i] - predicted_torque[i] for i in range(3)
        ]
        force_sources[source]["predicted_force_frame_n"] = predicted_frame
        force_sources[source]["corrected_torque_frame_nm"] = corrected_torque

    return {
        "fit_source": source,
        "fit_mode": mode,
        "joint_deg": reading["joint_deg"],
        "pose": pose,
        "force_sources": force_sources,
        "measured_force_frame_n": measured_frame,
        "predicted_force_frame_n": predicted_frame,
        "measured_force_base_n": measured_base,
        "measured_force_norm_n": _norm(measured_base),
        "corrected_force_base_n": corrected_base,
        "corrected_force_norm_n": _norm(corrected_base),
        "warnings": warnings,
    }


class SharedState:
    def __init__(self, initial: dict[str, object]):
        self._lock = threading.Lock()
        self._state = initial

    def update(self, state: dict[str, object]) -> None:
        with self._lock:
            self._state = state

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._state)


def _make_handler(shared: SharedState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/state.json"):
                payload = json.dumps(shared.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            payload = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only live RM65 force visualizer. It continuously reads current "
            "state/force, applies a fitted correction, and serves a base-frame "
            "3D force dashboard on localhost."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--fit-json", default="")
    parser.add_argument("--tool-label", default="")
    parser.add_argument("--expected-tool", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--warn-force-n", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rate_hz <= 0.0:
        print("ERROR: --rate-hz must be positive", file=sys.stderr)
        return 2
    if args.warn_force_n <= 0.0:
        print("ERROR: --warn-force-n must be positive", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    tool_label = args.tool_label or args.expected_tool or None
    fit_path = Path(args.fit_json) if args.fit_json else _default_fit_path(side, tool_label)
    try:
        fit_payload = _load_fit(fit_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    shared = SharedState(
        {
            "arm": side,
            "ip": ip,
            "timestamp": time.time(),
            "sequence": 0,
            "fit_json": str(fit_path),
            "fit_source": fit_payload["fit"].get("source"),
            "fit_mode": fit_payload["fit"].get("mode"),
            "warn_force_n": args.warn_force_n,
            "warnings": ["waiting for first robot reading"],
        }
    )
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(shared))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f"http://{args.host}:{args.port}/"

    print(f"RM65 live force visualizer: {url}")
    print(f"READ-ONLY: connected arm state and force reads only.")
    print(f"fit_json={fit_path}")
    print(f"fit_source={fit_payload['fit'].get('source')} mode={fit_payload['fit'].get('mode')}")

    stop_requested = False

    def _request_stop(signum, frame) -> None:
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, _request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, _request_stop)
    robot = None
    sequence = 0
    try:
        robot = Arm(RM65, ip)
        sample_interval = 1.0 / args.rate_hz
        while not stop_requested:
            loop_start = time.monotonic()
            reading = _read_once(robot, side)
            corrected = _apply_correction(reading, fit_payload)
            sequence += 1
            state = {
                "arm": side,
                "ip": ip,
                "timestamp": time.time(),
                "sequence": sequence,
                "fit_json": str(fit_path),
                "warn_force_n": args.warn_force_n,
                **corrected,
            }
            shared.update(state)
            if sequence == 1 or sequence % max(1, int(args.rate_hz)) == 0:
                print(
                    f"seq={sequence} corrected_base={_format(state['corrected_force_base_n'])} "
                    f"|F|={state['corrected_force_norm_n']:.4f}N"
                )
            elapsed = time.monotonic() - loop_start
            if elapsed < sample_interval:
                time.sleep(sample_interval - elapsed)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if robot is not None:
            robot.Arm_Socket_Close()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
