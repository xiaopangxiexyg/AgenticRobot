(function () {
  "use strict";

  var main = document.querySelector('main[data-resizable-sidebars="true"]');
  var handle = document.getElementById("sidebarResizeHandle");
  if (!main || !handle) return;

  var MIN_LEFT_PX = 260;
  var MIN_RIGHT_PX = 280;
  var KEYBOARD_STEP_PX = 24;
  var DEFAULT_LEFT_RATIO = 0.60;
  var dragging = false;

  function availableWidth() {
    var rect = main.getBoundingClientRect();
    var style = window.getComputedStyle(main);
    var columnGap = parseFloat(style.columnGap || style.gap || "0") || 0;
    return Math.max(0, rect.width - handle.offsetWidth - (2 * columnGap));
  }

  function limits() {
    var available = availableWidth();
    if (available <= 0) return { min: 0, max: 0 };
    var min = Math.min(MIN_LEFT_PX, Math.max(0, available - MIN_RIGHT_PX));
    var max = Math.max(min, available - Math.min(MIN_RIGHT_PX, available));
    return { min: min, max: max };
  }

  function currentLeftWidth() {
    var leftPanel = main.querySelector(":scope > .panel");
    return leftPanel ? leftPanel.getBoundingClientRect().width : 0;
  }

  function setLeftWidth(widthPx) {
    var bounds = limits();
    if (bounds.max <= 0) return;
    var next = Math.max(bounds.min, Math.min(bounds.max, Number(widthPx) || bounds.min));
    main.style.setProperty("--sidebar-left-width", next.toFixed(1) + "px");
    var percentage = Math.round((next / Math.max(1, availableWidth())) * 100);
    handle.setAttribute("aria-valuenow", String(Math.max(0, Math.min(100, percentage))));
  }

  function setFromPointer(event) {
    var rect = main.getBoundingClientRect();
    var style = window.getComputedStyle(main);
    var columnGap = parseFloat(style.columnGap || style.gap || "0") || 0;
    setLeftWidth(event.clientX - rect.left - columnGap);
  }

  function resetWidth() {
    setLeftWidth(availableWidth() * DEFAULT_LEFT_RATIO);
  }

  handle.addEventListener("pointerdown", function (event) {
    if (event.button !== 0) return;
    dragging = true;
    handle.classList.add("dragging");
    handle.setPointerCapture(event.pointerId);
    setFromPointer(event);
    event.preventDefault();
  });

  handle.addEventListener("pointermove", function (event) {
    if (!dragging) return;
    setFromPointer(event);
  });

  function endDrag(event) {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    if (handle.hasPointerCapture(event.pointerId)) {
      handle.releasePointerCapture(event.pointerId);
    }
  }

  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);
  handle.addEventListener("dblclick", resetWidth);
  handle.addEventListener("keydown", function (event) {
    if (event.key === "ArrowLeft") {
      setLeftWidth(currentLeftWidth() - KEYBOARD_STEP_PX);
    } else if (event.key === "ArrowRight") {
      setLeftWidth(currentLeftWidth() + KEYBOARD_STEP_PX);
    } else if (event.key === "Home") {
      resetWidth();
    } else {
      return;
    }
    event.preventDefault();
  });

  window.addEventListener("resize", function () {
    if (window.matchMedia("(max-width: 1100px)").matches) return;
    setLeftWidth(currentLeftWidth());
  });
})();
