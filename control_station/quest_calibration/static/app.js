(function () {
  "use strict";

  var clientId = "quest-calibration-ui-" + Math.random().toString(36).slice(2);
  var clientSequence = 0;
  var latest = null;
  var windowCounter = 1;
  var requestInFlight = false;
  var drainInProgress = false;

  function byId(id) { return document.getElementById(id); }
  function value(id) { return String(byId(id).value || "").trim(); }

  function setDisabled(id, disabled) {
    byId(id).disabled = Boolean(disabled);
  }

  function nextWindowId() {
    return "stationary-window-" + String(windowCounter).padStart(3, "0");
  }

  function operatorStatus() {
    return latest && latest.operator ? latest.operator : null;
  }

  function render(payload) {
    latest = payload || {};
    var operator = operatorStatus();
    byId("bindingState").textContent = latest.binding_state || "unknown";
    byId("captureState").textContent = operator ? operator.state : "-";
    byId("completedCount").textContent = operator ? operator.completed_window_count : 0;
    byId("status").textContent = JSON.stringify(payload, null, 2);

    var candidates = operator && Array.isArray(operator.candidates) ? operator.candidates : [];
    var candidateSelect = byId("candidate");
    var selected = candidateSelect.value;
    candidateSelect.innerHTML = "";
    candidates.forEach(function (candidate) {
      var option = document.createElement("option");
      option.value = candidate.candidate_hash;
      option.textContent = candidate.handedness + " / " + candidate.reference_space_type + " / " + candidate.candidate_hash.slice(0, 16);
      candidateSelect.appendChild(option);
    });
    if (selected && candidates.some(function (item) { return item.candidate_hash === selected; })) candidateSelect.value = selected;

    var actions = operator && operator.actions ? operator.actions : {};
    setDisabled("selectSource", requestInFlight || drainInProgress || !actions.can_select_source);
    setDisabled("beginWindow", requestInFlight || drainInProgress || !actions.can_begin_window);
    setDisabled("completeWindow", requestInFlight || drainInProgress || !actions.can_complete_window);
    setDisabled("seal", requestInFlight || drainInProgress || !actions.can_seal);
    setDisabled("cancel", requestInFlight || !actions.can_cancel);
  }

  function refresh() {
    return fetch("/status.json", {cache: "no-store"})
      .then(function (response) { return response.json(); })
      .then(render)
      .catch(function (error) { byId("status").textContent = "status error: " + error; });
  }

  function submit(kind, extra) {
    var operator = operatorStatus();
    clientSequence += 1;
    var payload = {
      client_id: clientId,
      client_sequence: clientSequence,
      expected_revision: operator ? operator.revision : 0,
      intent_kind: kind
    };
    Object.keys(extra || {}).forEach(function (key) { payload[key] = extra[key]; });
    requestInFlight = true;
    render(latest);
    return fetch("/intent", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    })
      .then(function (response) { return response.json(); })
      .then(function (receipt) {
        byId("status").textContent = JSON.stringify(receipt, null, 2);
        return refresh().then(function () { return receipt; });
      })
      .catch(function (error) {
        byId("status").textContent = "intent error: " + error;
        return {accepted: false, rejection_code: "request_failed"};
      })
      .then(function (receipt) {
        requestInFlight = false;
        render(latest);
        return receipt;
      });
  }

  function completeWindowWithBoundedDrain(attempt) {
    submit("complete_window").then(function (receipt) {
      if (receipt.accepted && receipt.outcome === "window_completed") {
        windowCounter += 1;
        byId("windowId").value = nextWindowId();
        drainInProgress = false;
        render(latest);
        return;
      }
      if (receipt.accepted && receipt.outcome === "window_drain_pending" && attempt < 20) {
        window.setTimeout(function () { completeWindowWithBoundedDrain(attempt + 1); }, 250);
        return;
      }
      drainInProgress = false;
      render(latest);
    });
  }

  byId("selectSource").onclick = function () {
    submit("select_source", {
      candidate_hash: value("candidate"), robot_side: value("robotSide"),
      reference_frame_id: value("referenceFrame"), moving_frame_id: value("movingFrame"),
      xr_frame_clock_domain_id: value("frameClock")
    });
  };
  byId("beginWindow").onclick = function () { submit("begin_window", {window_id: value("windowId")}); };
  byId("completeWindow").onclick = function () {
    if (drainInProgress) return;
    drainInProgress = true;
    completeWindowWithBoundedDrain(1);
  };
  byId("seal").onclick = function () { submit("seal"); };
  byId("cancel").onclick = function () { submit("cancel", {reason: "operator_web_cancel"}); };

  byId("windowId").value = nextWindowId();
  render({binding_state: "loading", operator: null});
  refresh();
  window.setInterval(refresh, 500);
})();
