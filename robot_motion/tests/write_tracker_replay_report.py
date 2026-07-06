from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

from .tracker_replay import (
    OfflineTrackerReplayConfig,
    OfflineTrackerReplayReport,
    run_offline_tracker_replay,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a portable HTML report for offline tracker replay comparison."
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot-style dataset root.")
    parser.add_argument("--episode", default="0", help="Episode index/id, e.g. 0 or episode_000000.")
    parser.add_argument("--start-frame", type=int, default=0, help="Sorted row offset to start from.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to replay; 0 means all.")
    parser.add_argument(
        "--active-sides",
        default="left,right",
        help="Comma-separated sides to compare. Default: left,right.",
    )
    parser.add_argument(
        "--initial-qpos-source",
        choices=("first_observation", "calibration_home", "default"),
        default="first_observation",
        help="Initial solver qpos for offline replay.",
    )
    parser.add_argument("--disable-target-smoothing", action="store_true")
    parser.add_argument("--disable-realistic-joint-step", action="store_true")
    parser.add_argument(
        "--solve-max-iterations",
        type=int,
        default=1,
        help="Dual-arm IK solve iterations per frame. Default matches runtime.",
    )
    parser.add_argument(
        "--seed-each-frame-from-observation",
        action="store_true",
        help="Reset solver qpos from the dataset observation qpos before each frame.",
    )
    parser.add_argument("--scene-profile", default="", help="Override collision scene profile.")
    parser.add_argument("--contact-mode", default="", help="Override collision avoidance mode.")
    parser.add_argument("--solver-frequency", type=float, default=0.0, help="Override dual-arm IK Hz.")
    parser.add_argument("--ik-tuning-json", default="", help="Optional IK tuning JSON override.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML report path. Default: <dataset>/visualizations/offline_tracker_replay_<episode>.html",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    active_sides = tuple(
        side.strip().lower()
        for side in str(args.active_sides).replace("both", "left,right").split(",")
        if side.strip()
    )
    report = run_offline_tracker_replay(
        OfflineTrackerReplayConfig(
            dataset_root=args.dataset_root,
            episode=args.episode,
            start_frame=int(args.start_frame),
            max_frames=int(args.max_frames),
            active_sides=active_sides,
            use_target_smoothing=not bool(args.disable_target_smoothing),
            enforce_realistic_joint_step=not bool(args.disable_realistic_joint_step),
            initial_qpos_source=str(args.initial_qpos_source),
            solve_max_iterations=int(args.solve_max_iterations),
            seed_each_frame_from_observation=bool(args.seed_each_frame_from_observation),
            scene_profile=str(args.scene_profile or "") or None,
            contact_mode=str(args.contact_mode or "") or None,
            solver_frequency=float(args.solver_frequency) if float(args.solver_frequency) > 0.0 else None,
            ik_tuning_json_path=str(args.ik_tuning_json or "") or None,
        )
    )
    output = args.output or (
        report.dataset_root
        / "visualizations"
        / f"offline_tracker_replay_{report.episode_id}.html"
    )
    path = write_offline_tracker_gui(report, output)
    print(path)
    return 0


def write_offline_tracker_gui(report: OfflineTrackerReplayReport, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict(include_frames=True)
    payload["video_sources"] = _video_sources_for_output(report, output)
    output.write_text(_html_document(payload), encoding="utf-8")
    return output.resolve()


def _video_sources_for_output(report: OfflineTrackerReplayReport, output_path: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    output_dir = output_path.expanduser().resolve().parent
    dataset_root = report.dataset_root.resolve()
    for key, relative in report.video_paths.items():
        video_path = (report.dataset_root / relative).resolve()
        try:
            output_dir.relative_to(dataset_root)
        except ValueError:
            sources[key] = video_path.as_uri()
        else:
            sources[key] = os.path.relpath(video_path, output_dir)
    return sources


def _html_document(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    title = f"Offline Tracker Replay - {payload.get('episode_id', '')}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f8fb;
  --ink: #16202a;
  --muted: #5f6b7a;
  --line: #d8dee8;
  --panel: #ffffff;
  --accent: #0b6bcb;
  --bad: #b42318;
  --ok: #12715b;
}}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
header {{
  padding: 14px 18px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  position: sticky;
  top: 0;
  z-index: 3;
}}
h1 {{
  font-size: 18px;
  margin: 0 0 8px;
  letter-spacing: 0;
}}
.sub {{
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}}
main {{
  display: grid;
  grid-template-columns: minmax(420px, 1.35fr) minmax(360px, 0.85fr);
  gap: 14px;
  padding: 14px;
}}
section {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 12px;
}}
.controls {{
  display: grid;
  grid-template-columns: auto 1fr auto auto;
  gap: 10px;
  align-items: center;
  margin-top: 10px;
}}
button {{
  border: 1px solid var(--line);
  background: #f2f5f9;
  color: var(--ink);
  border-radius: 5px;
  padding: 7px 10px;
  cursor: pointer;
}}
input[type=range] {{
  width: 100%;
}}
.videos {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}}
video {{
  width: 100%;
  background: #000;
  border-radius: 4px;
  aspect-ratio: 4 / 3;
}}
.video-label {{
  color: var(--muted);
  font-size: 12px;
  margin: 4px 0 0;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px;
}}
.metric {{
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 8px;
}}
.metric strong {{
  display: block;
  font-size: 12px;
  color: var(--muted);
  font-weight: 600;
}}
.metric span {{
  font-size: 18px;
}}
canvas {{
  width: 100%;
  height: 220px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: #fff;
}}
table {{
  width: 100%;
  border-collapse: collapse;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 6px 4px;
  text-align: left;
  vertical-align: top;
}}
th {{
  color: var(--muted);
  font-weight: 600;
}}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}}
.ok {{ color: var(--ok); }}
.bad {{ color: var(--bad); }}
@media (max-width: 980px) {{
  main {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="sub" id="subtitle"></div>
  <div class="controls">
    <button id="play">Play</button>
    <input id="frame" type="range" min="0" max="0" value="0">
    <span id="frameLabel"></span>
    <select id="side"></select>
  </div>
</header>
<main>
  <div>
    <section>
      <h2>Videos</h2>
      <div class="videos" id="videos"></div>
    </section>
    <section>
      <h2>Error Curves</h2>
      <canvas id="chart" width="900" height="260"></canvas>
    </section>
  </div>
  <div>
    <section>
      <h2>Summary</h2>
      <div class="metrics" id="summary"></div>
    </section>
    <section>
      <h2>Frame</h2>
      <table id="frameTable"></table>
    </section>
  </div>
</main>
<script>
const report = {data_json};
const frames = report.frames || [];
const sides = report.active_sides || ["left", "right"];
const videoSources = report.video_sources || {{}};
let playing = false;
let timer = null;

function fmt(value, digits=4) {{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}}

function metricMax(side, name) {{
  const bySide = report.summary?.by_side || {{}};
  return bySide[side]?.[name]?.max;
}}

function init() {{
  document.getElementById("subtitle").innerHTML = [
    `<span><code>${{escapeHtml(report.dataset_root || "")}}</code></span>`,
    `<span>episode=${{escapeHtml(report.episode_id || "")}}</span>`,
    `<span>frames=${{report.frame_count || 0}}/${{report.source_frame_count || 0}}</span>`,
    `<span>hz=${{fmt(report.nominal_hz, 3)}}</span>`,
    `<span>scene=${{escapeHtml(report.summary?.scene_profile || "")}}</span>`
  ].join("");
  const slider = document.getElementById("frame");
  slider.max = Math.max(0, frames.length - 1);
  slider.addEventListener("input", () => setFrame(Number(slider.value), true));
  const sideSelect = document.getElementById("side");
  sides.forEach(side => {{
    const option = document.createElement("option");
    option.value = side;
    option.textContent = side;
    sideSelect.appendChild(option);
  }});
  sideSelect.addEventListener("change", () => drawChart(Number(slider.value)));
  document.getElementById("play").addEventListener("click", togglePlay);
  initVideos();
  renderSummary();
  setFrame(0, true);
}}

function initVideos() {{
  const root = document.getElementById("videos");
  Object.entries(videoSources).forEach(([key, src]) => {{
    const wrap = document.createElement("div");
    const video = document.createElement("video");
    video.src = src;
    video.controls = true;
    video.muted = true;
    video.preload = "metadata";
    video.dataset.key = key;
    const label = document.createElement("div");
    label.className = "video-label";
    label.textContent = key;
    wrap.appendChild(video);
    wrap.appendChild(label);
    root.appendChild(wrap);
  }});
}}

function renderSummary() {{
  const root = document.getElementById("summary");
  root.innerHTML = "";
  sides.forEach(side => {{
    [
      ["source pos max m", metricMax(side, "source_vs_action_translation_error_m"), 5],
      ["ik pos max m", metricMax(side, "ik_vs_action_translation_error_m"), 5],
      ["ik rot max deg", metricMax(side, "ik_vs_action_rotation_error_deg"), 3],
      ["qpos max deg", metricMax(side, "qpos_max_abs_error_deg"), 3],
    ].forEach(([label, value, digits]) => {{
      const item = document.createElement("div");
      item.className = "metric";
      item.innerHTML = `<strong>${{escapeHtml(side)}} ${{label}}</strong><span>${{fmt(value, digits)}}</span>`;
      root.appendChild(item);
    }});
  }});
}}

function setFrame(index, syncVideo) {{
  if (!frames.length) return;
  const clamped = Math.max(0, Math.min(frames.length - 1, index));
  document.getElementById("frame").value = clamped;
  const frame = frames[clamped];
  document.getElementById("frameLabel").textContent =
    `${{clamped + 1}}/${{frames.length}} frame=${{frame.frame_index}} t=${{fmt(frame.frame_time_sec, 3)}}s`;
  if (syncVideo) syncVideos(frame.frame_time_sec || 0);
  renderFrame(frame);
  drawChart(clamped);
}}

function syncVideos(timeSec) {{
  document.querySelectorAll("video").forEach(video => {{
    if (!Number.isFinite(video.duration)) return;
    const target = Math.max(0, Math.min(video.duration, timeSec));
    if (Math.abs(video.currentTime - target) > 0.08) video.currentTime = target;
  }});
}}

function renderFrame(frame) {{
  const side = document.getElementById("side").value || sides[0];
  const err = frame.errors?.[side] || {{}};
  const rows = [
    ["status", frame.status || ""],
    ["fallback", frame.fallback_reason || ""],
    ["ik pos err m", fmt(err.translation_error_m, 6)],
    ["ik rot err deg", fmt(err.rotation_error_deg, 4)],
    ["source pos err m", fmt(frame.source_vs_action_errors?.[side]?.translation_error_m, 6)],
    ["source rot err deg", fmt(frame.source_vs_action_errors?.[side]?.rotation_error_deg, 4)],
    ["qpos max deg", fmt(err.qpos_max_abs_error_deg, 4)],
    ["source desk xyz", xyz(frame.source_pose_desk?.[side])],
    ["original action xyz", xyz(frame.original_action_pose_base?.[side])],
    ["replay action xyz", xyz(frame.replay_action_pose_base?.[side])],
    ["original qpos", vec(frame.original_qpos?.[side])],
    ["replay qpos", vec(frame.replay_qpos?.[side])],
  ];
  document.getElementById("frameTable").innerHTML = rows.map(([k, v]) =>
    `<tr><th>${{escapeHtml(k)}}</th><td><code>${{escapeHtml(String(v))}}</code></td></tr>`
  ).join("");
}}

function drawChart(currentIndex) {{
  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const y = 20 + i * (h - 40) / 4;
    ctx.beginPath();
    ctx.moveTo(42, y);
    ctx.lineTo(w - 12, y);
    ctx.stroke();
  }}
  const side = document.getElementById("side").value || sides[0];
  const pos = frames.map(f => f.errors?.[side]?.translation_error_m ?? null);
  const rot = frames.map(f => f.errors?.[side]?.rotation_error_deg ?? null);
  drawSeries(ctx, pos, "#0b6bcb", w, h, "pos m");
  drawSeries(ctx, rot, "#b42318", w, h, "rot deg");
  const x = 42 + currentIndex * (w - 54) / Math.max(1, frames.length - 1);
  ctx.strokeStyle = "#16202a";
  ctx.beginPath();
  ctx.moveTo(x, 14);
  ctx.lineTo(x, h - 18);
  ctx.stroke();
}}

function drawSeries(ctx, values, color, w, h, label) {{
  const finite = values.filter(v => v !== null && Number.isFinite(Number(v))).map(Number);
  if (!finite.length) return;
  const max = Math.max(...finite, 1e-9);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((raw, i) => {{
    if (raw === null || !Number.isFinite(Number(raw))) return;
    const x = 42 + i * (w - 54) / Math.max(1, values.length - 1);
    const y = (h - 20) - Number(raw) / max * (h - 46);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(`${{label}} max=${{fmt(max, 4)}}`, 48, color === "#0b6bcb" ? 18 : 34);
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById("play").textContent = playing ? "Pause" : "Play";
  if (timer) clearInterval(timer);
  if (!playing) return;
  timer = setInterval(() => {{
    const slider = document.getElementById("frame");
    const next = Number(slider.value) + 1;
    if (next >= frames.length) {{
      playing = false;
      document.getElementById("play").textContent = "Play";
      clearInterval(timer);
      return;
    }}
    setFrame(next, true);
  }}, 1000 / Math.max(1, Number(report.nominal_hz || 15)));
}}

function xyz(values) {{
  if (!values) return "n/a";
  return values.slice(0, 3).map(v => fmt(v, 5)).join(", ");
}}

function vec(values) {{
  if (!values) return "n/a";
  return values.map(v => fmt(v, 4)).join(", ");
}}

function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, ch => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }}[ch]));
}}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
