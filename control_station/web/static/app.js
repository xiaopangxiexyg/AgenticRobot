"use strict";

let latestState = null;
let clientSequence = 0;
let latestEventSequence = -1;
let recentEvents = [];
const clientId = `browser:${crypto.randomUUID()}`;

function text(value) {
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

function fillDefinitionList(elementId, rows) {
  const root = document.getElementById(elementId);
  root.replaceChildren();
  for (const [label, value] of rows) {
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = text(value);
    root.append(term, detail);
  }
}

function syncIntentOptions(allowedKinds) {
  const select = document.getElementById("intent-kind");
  const normalized = Array.isArray(allowedKinds) ? allowedKinds.map(String) : [];
  const currentKinds = Array.from(select.options, (option) => option.value);
  if (JSON.stringify(currentKinds) === JSON.stringify(normalized)) return;
  const prior = select.value;
  select.replaceChildren();
  for (const kind of normalized) {
    const option = document.createElement("option");
    option.value = kind;
    option.textContent = kind;
    select.append(option);
  }
  if (normalized.includes(prior)) select.value = prior;
}

function render(state) {
  latestState = state;
  fillDefinitionList("run-state", [
    ["Lifecycle", state.station.lifecycle_state],
    ["Revision", state.station.state_revision],
    ["Session", state.runtime_session_id],
    ["Domain", state.control_domain_id],
    ["Manifest", state.manifest.manifest_id],
    ["Manifest hash", state.manifest.semantic_content_hash],
  ]);

  const sources = document.getElementById("source-list");
  sources.replaceChildren();
  for (const source of state.sources) {
    const card = document.createElement("article");
    card.className = "source-card";
    const title = document.createElement("h3");
    title.textContent = source.source_slot_id;
    const kind = document.createElement("p");
    kind.textContent = `${source.source_kind} · ${source.runtime_state}`;
    const seam = document.createElement("code");
    seam.textContent = source.expected_emission_seam;
    card.append(title, kind, seam);
    sources.append(card);
  }

  fillDefinitionList("target-state", [
    ["Binding", state.target.binding_id],
    ["Kind", state.target.target_kind],
    ["Runtime", state.target.runtime_state],
    ["Embodiment", state.target.embodiment_instance_id],
    ["Motion", state.target.motion_authorized ? "AUTHORIZED" : "DENIED"],
  ]);

  fillDefinitionList("authority-state", [
    ["Dispatch owner", state.control_facts.dispatch_owner],
    ["Authority epoch", state.control_facts.authority_epoch],
    ["Execution epoch", state.control_facts.execution_epoch],
    ["Operator lease", state.operator_control_lease?.client_id],
    ["Authorization status", state.target.authorization_status],
    ["Publication", state.control_facts.station_publication?.publication_enabled],
    ["Publication generation", state.control_facts.station_publication?.publication_generation],
  ]);
  document.getElementById("trace-state").textContent = JSON.stringify(
    {
      pending_operations: state.pending_operations,
      capture_status: state.control_facts.capture_status,
      capabilities: state.capabilities,
      recent_ui_events: recentEvents,
    },
    null,
    2,
  );

  const intentEnabled = state.capabilities.intent_submission === true;
  const coordinated = state.capabilities.operation_coordinator === true;
  const scopedSimAuthority = state.maturity === "cs2a2b_scoped_sim_authority_api";
  document.getElementById("maturity").textContent = scopedSimAuthority
    ? "CS-2a2b · SCOPED SIM AUTHORITY · NO ACTION PAYLOAD"
    : (coordinated
      ? "CS-1c · NO-I/O COORDINATION · NO MOTION"
      : (intentEnabled ? "CS-1b · TYPED INTENTS · NO MOTION" : "CS-1a · READ ONLY"));
  document.getElementById("intent-boundary").textContent = scopedSimAuthority
    ? "仅可提交server发布的typed lifecycle/candidate/authority intent；HTTP token只授权本地API，adapter completion和publication barrier仍独立生效。"
    : (intentEnabled
      ? (coordinated
        ? "白名单intent只进入有界no-I/O coordinator；API token不是operator lease或motion authority。"
        : "仅允许白名单station intent；API token不是operator lease或motion authority。")
      : "当前server为只读模式，intent submission不可用。");
  document.getElementById("authority-boundary").textContent = scopedSimAuthority
    ? "此页不接受动作数值。SIM publication仅在lease、preflight、candidate与typed authority completion全部通过后开放；real target被builder拒绝。"
    : "Web API token和operator lease都不构成motion authorization；CS-1a/1b/1c均禁止motion。";
  syncIntentOptions(state.capabilities.allowed_intent_kinds);
  for (const id of ["access-token", "intent-kind", "intent-subject", "intent-submit"]) {
    document.getElementById(id).disabled = !intentEnabled;
  }
}

async function refreshEvents() {
  if (!latestState?.capabilities?.event_fanout) return;
  const response = await fetch(`/api/events?after_sequence=${latestEventSequence}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`event HTTP ${response.status}`);
  const snapshot = await response.json();
  if (snapshot.gap_detected) recentEvents = [];
  recentEvents.push(...snapshot.events);
  recentEvents = recentEvents.slice(-50);
  if (snapshot.latest_sequence !== null) latestEventSequence = snapshot.latest_sequence;
}

async function refresh() {
  const health = document.getElementById("health");
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    await refreshEvents();
    render(latestState);
    if (latestState.maturity === "cs2a2b_scoped_sim_authority_api") {
      health.textContent = latestState.target.motion_authorized
        ? "SIM PUBLICATION ACTIVE"
        : "SCOPED SIM · DENIED";
    } else {
      health.textContent = latestState.capabilities.intent_submission ? "NO MOTION" : "READ ONLY";
    }
    health.className = "badge good";
  } catch (error) {
    health.textContent = `OFFLINE · ${error.message}`;
    health.className = "badge bad";
  }
}

refresh();
window.setInterval(refresh, 1000);

document.getElementById("intent-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!latestState?.capabilities?.intent_submission) return;
  const token = document.getElementById("access-token").value;
  const intentKind = document.getElementById("intent-kind").value;
  const subjectId = document.getElementById("intent-subject").value.trim();
  const payload = {
    client_id: clientId,
    client_sequence: clientSequence,
    intent_kind: intentKind,
    produced_at: performance.now() / 1000,
    produced_clock_domain_id: "browser-performance",
    expected_state_revision: latestState.station.state_revision,
    manifest_hash: latestState.manifest.semantic_content_hash,
    reason: "local_control_station_gui",
  };
  if (subjectId) payload.subject_id = subjectId;

  const result = document.getElementById("intent-result");
  try {
    const response = await fetch("/api/intents", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    result.textContent = JSON.stringify(body, null, 2);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    clientSequence += 1;
    latestState = body.station_state;
    render(latestState);
  } catch (error) {
    result.textContent = `${result.textContent}\n${error.message}`;
  }
});
