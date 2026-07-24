"""Dependency-free run recording and a self-contained HTML progress report."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


METRIC_COLUMNS = (
    "update",
    "global_step",
    "wall_time_seconds",
    "transitions_per_second",
    "raw_reward_mean",
    "shaped_reward_mean",
    "recent_episode_raw_return_mean",
    "completed_episodes",
    "successful_episodes",
    "success_rate",
    "rollout_completed_episodes",
    "rollout_successful_episodes",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "policy_std_mean",
    "policy_log_std_mean",
    "policy_action_dim",
    "environment_action_dim",
    "learning_rate",
    "safety_projection_rate",
    "safety_rejection_rate",
    "safety_recovery_rate",
    "safety_projection_alpha_mean",
    "safety_minimum_distance_m",
    "hard_penetration_count",
    "clearance_violation_count",
    "contact_loss_termination_count",
    "arm_tracking_error_rad_mean",
    "arm_tracking_error_rad_max",
    "gripper_tracking_error_mean",
    "gripper_tracking_error_max",
    "wrist_rgb_mean",
    "wrist_rgb_std",
    "wrist_rgb_health_rate",
    "reward_distance_m_mean",
    "reward_grasp_distance_m_mean",
    "reward_horizontal_distance_m_mean",
    "reward_vertical_error_m_mean",
    "reward_grasp_gate_mean",
    "reward_close_gate_mean",
    "reward_reach_phase_weight_mean",
    "reward_gripper_close_mean",
    "reward_grasp_close_confidence_mean",
    "reward_grasp_acquire_term_mean",
    "reward_bilateral_contact_term_mean",
    "reward_bilateral_contact_acquire_term_mean",
    "reward_bilateral_contact_missing_term_mean",
    "reward_grasp_latched_mean",
    "reward_lift_fraction_mean",
    "reward_reach_term_mean",
    "reward_reach_progress_term_mean",
    "reward_grasp_region_term_mean",
    "reward_gripper_term_mean",
    "reward_gripper_progress_term_mean",
    "reward_latch_term_mean",
    "reward_premature_close_term_mean",
    "reward_lift_term_mean",
    "reward_lift_progress_term_mean",
    "reward_action_term_mean",
    "reward_success_term_mean",
    "reward_safety_projection_term_mean",
)


class RunRecorder:
    def __init__(
        self,
        root: str | Path,
        *,
        run_name: str | None,
        config: Mapping[str, Any],
    ) -> None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        suffix = _safe_name(run_name) if run_name else "ppo"
        base = Path(root) / f"{timestamp}_{suffix}"
        self.directory = _unique_directory(base)
        self.directory.mkdir(parents=True)
        self.checkpoint_directory = self.directory / "checkpoints"
        self.checkpoint_directory.mkdir()
        self.metrics_jsonl = self.directory / "metrics.jsonl"
        self.metrics_csv = self.directory / "metrics.csv"
        self.episodes_jsonl = self.directory / "episodes.jsonl"
        self.summary_json = self.directory / "summary.json"
        self.report_html = self.directory / "report.html"
        self._metrics: list[dict[str, Any]] = []
        _write_json_atomic(self.directory / "run_config.json", dict(config))

    def record_update(
        self,
        metrics: Mapping[str, Any],
        episodes: Sequence[Mapping[str, Any]],
    ) -> None:
        record = _json_safe_mapping(metrics)
        self._metrics.append(record)
        _append_json_line(self.metrics_jsonl, record)
        csv_exists = self.metrics_csv.exists()
        with self.metrics_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
            if not csv_exists:
                writer.writeheader()
            writer.writerow({name: record.get(name) for name in METRIC_COLUMNS})
        for episode in episodes:
            _append_json_line(self.episodes_jsonl, _json_safe_mapping(episode))
        self.write_report()

    def finish(self, summary: Mapping[str, Any]) -> None:
        _write_json_atomic(self.summary_json, _json_safe_mapping(summary))
        self.write_report(summary=summary)

    def write_report(self, *, summary: Mapping[str, Any] | None = None) -> None:
        config = json.loads((self.directory / "run_config.json").read_text(encoding="utf-8"))
        embedded = json.dumps(self._metrics, ensure_ascii=False).replace("</", "<\\/")
        summary_data = {} if summary is None else _json_safe_mapping(summary)
        summary_embedded = json.dumps(summary_data, ensure_ascii=False).replace("</", "<\\/")
        config_embedded = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
        html = _REPORT_TEMPLATE.replace("__METRICS__", embedded)
        html = html.replace("__SUMMARY__", summary_embedded)
        html = html.replace("__CONFIG__", config_embedded)
        temporary = self.report_html.with_suffix(".html.tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(self.report_html)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError("run_name must contain at least one safe character.")
    return cleaned[:80]


def _unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique run directory below {path.parent}.")


def _json_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    # Round-trip catches accidental tensors/arrays before a long run creates an
    # unreadable partial log.
    return json.loads(json.dumps(dict(values), ensure_ascii=False, allow_nan=False))


def _append_json_line(path: Path, values: Mapping[str, Any]) -> None:
    line = json.dumps(dict(values), ensure_ascii=False, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _write_json_atomic(path: Path, values: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(values), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_REPORT_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Cube-Lift PPO</title>
<style>
body{font:14px system-ui,sans-serif;margin:24px;background:#0d1117;color:#c9d1d9}
h1,h2{color:#f0f6fc}.cards{display:flex;flex-wrap:wrap;gap:10px}.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;min-width:150px}
.value{font-size:22px;color:#58a6ff}canvas{background:#fff;border-radius:8px;width:100%;max-width:1100px;height:300px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px;border-bottom:1px solid #30363d;text-align:right}th:first-child,td:first-child{text-align:left}details{margin:16px 0}pre{white-space:pre-wrap;background:#161b22;padding:12px;border-radius:8px}
</style></head><body><h1>Cube-Lift temporary PPO</h1><div id="cards" class="cards"></div>
<h2>Reward / success</h2><canvas id="reward" width="1100" height="300"></canvas>
<h2>PPO losses</h2><canvas id="loss" width="1100" height="300"></canvas>
<h2>最近 updates</h2><div id="table"></div><details><summary>Run config</summary><pre id="config"></pre></details>
<script>const metrics=__METRICS__,summary=__SUMMARY__,config=__CONFIG__;
const latest=metrics.at(-1)||{};const cards=[['step',latest.global_step],['update',latest.update],['TPS',latest.transitions_per_second],['success',latest.success_rate],['reach distance',latest.reward_distance_m_mean],['gripper close',latest.reward_gripper_close_mean],['lift fraction',latest.reward_lift_fraction_mean],['projection rate',latest.safety_projection_rate],['contact loss',latest.contact_loss_termination_count],['clearance violation',latest.clearance_violation_count],['hard penetration',latest.hard_penetration_count],['raw return',latest.recent_episode_raw_return_mean],['policy std',latest.policy_std_mean]];
document.getElementById('cards').innerHTML=cards.map(([k,v])=>`<div class="card"><div>${k}</div><div class="value">${typeof v==='number'?v.toFixed(4):v??'-'}</div></div>`).join('');
function chart(id,series){const c=document.getElementById(id),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);const all=series.flatMap(s=>metrics.map(m=>m[s[1]]).filter(Number.isFinite));if(!all.length)return;let lo=Math.min(...all),hi=Math.max(...all);if(lo===hi){lo-=1;hi+=1}x.strokeStyle='#ddd';x.strokeRect(45,10,c.width-55,c.height-35);series.forEach((s,j)=>{x.strokeStyle=s[2];x.beginPath();metrics.forEach((m,i)=>{const v=m[s[1]];if(!Number.isFinite(v))return;const px=45+i/Math.max(metrics.length-1,1)*(c.width-55),py=10+(hi-v)/(hi-lo)*(c.height-35);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle=s[2];x.fillText(s[0],55+j*170,25)});x.fillStyle='#222';x.fillText(hi.toFixed(3),2,18);x.fillText(lo.toFixed(3),2,c.height-25)}
chart('reward',[['raw reward','raw_reward_mean','#0969da'],['shaped reward','shaped_reward_mean','#1a7f37'],['recent return','recent_episode_raw_return_mean','#bf8700'],['success rate','success_rate','#cf222e']]);chart('loss',[['policy loss','policy_loss','#8250df'],['value loss','value_loss','#cf222e'],['KL','approx_kl','#0969da']]);
const cols=['update','global_step','shaped_reward_mean','success_rate','reward_distance_m_mean','reward_grasp_distance_m_mean','reward_grasp_gate_mean','reward_gripper_close_mean','reward_grasp_latched_mean','reward_lift_fraction_mean','reward_reach_term_mean','reward_grasp_region_term_mean','reward_gripper_term_mean','reward_latch_term_mean','reward_lift_term_mean','safety_projection_rate','contact_loss_termination_count','clearance_violation_count','hard_penetration_count','policy_loss','value_loss','approx_kl'];const rows=metrics.slice(-20).reverse();document.getElementById('table').innerHTML=`<table><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${typeof r[c]==='number'?r[c].toFixed(5):r[c]??''}</td>`).join('')}</tr>`).join('')}</tbody></table>`;document.getElementById('config').textContent=JSON.stringify(config,null,2);</script></body></html>"""


__all__ = ["METRIC_COLUMNS", "RunRecorder"]
