(function () {
  "use strict";

  var lastSequence = 0;
  var timers = Object.create(null);
  var pulseMs = 650;

  function cell(token) {
    return document.querySelector('[data-input-activity="' + token + '"]');
  }

  function flash(token) {
    var target = cell(token);
    if (!target) return;
    target.classList.remove("active");
    // Force a fresh transition for rapid repeated presses of the same input.
    void target.offsetWidth;
    target.classList.add("active");
    if (timers[token]) window.clearTimeout(timers[token]);
    timers[token] = window.setTimeout(function () {
      target.classList.remove("active");
      timers[token] = null;
    }, pulseMs);
  }

  function editable(target) {
    if (!target) return false;
    var tag = String(target.tagName || "").toUpperCase();
    return Boolean(
      target.isContentEditable
      || ["INPUT", "TEXTAREA", "SELECT"].indexOf(tag) >= 0
    );
  }

  document.addEventListener("keydown", function (event) {
    if (event.repeat || editable(event.target)) return;
    var key = String(event.key || "").toLowerCase();
    if (["a", "b", "c", "g"].indexOf(key) >= 0) flash("key_" + key);
  }, true);

  window.dexTeleopRenderInputActivity = function (state) {
    var activity = state && state.input_activity;
    if (!activity) return;
    var events = Array.isArray(activity.events) ? activity.events : [activity];
    events.forEach(function (entry) {
      var sequence = Number(entry && entry.sequence || 0);
      if (!Number.isFinite(sequence) || sequence <= lastSequence) return;
      lastSequence = sequence;
      if (String(entry.source || "") !== "foot_pad") return;
      var index = Number(entry.pedal_index || 0);
      if (Number.isInteger(index) && index >= 1 && index <= 5) {
        flash("pedal_" + String(index));
      }
    });
    var publishedSequence = Number(activity.sequence || 0);
    if (Number.isFinite(publishedSequence) && publishedSequence > lastSequence) {
      lastSequence = publishedSequence;
    }
  };
})();
