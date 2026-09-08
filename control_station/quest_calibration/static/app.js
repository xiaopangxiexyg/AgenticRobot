(function () {
  "use strict";

  function byId(id) { return document.getElementById(id); }

  function render(payload) {
    payload = payload || {};
    var operator = payload.operator || null;
    byId("bindingState").textContent = payload.binding_state || "unknown";
    byId("captureState").textContent = operator ? operator.state : "-";
    byId("completedCount").textContent = operator ? operator.completed_window_count : 0;
    var candidates = operator && Array.isArray(operator.candidates) ? operator.candidates : [];
    var candidateText = candidates.length === 0
      ? "等待 Quest 进入 MR/VR"
      : candidates.map(function (candidate) {
        return candidate.handedness + " / " + candidate.reference_space_type + " / " + candidate.candidate_hash.slice(0, 16);
      }).join(", ");
    byId("candidateSummary").textContent = candidateText;
    byId("status").textContent = JSON.stringify(payload, null, 2);
  }

  function refresh() {
    return fetch("/status.json", {cache: "no-store"})
      .then(function (response) { return response.json(); })
      .then(render)
      .catch(function (error) { byId("status").textContent = "status error: " + error; });
  }

  render({binding_state: "loading", operator: null});
  refresh();
  window.setInterval(refresh, 500);
})();
