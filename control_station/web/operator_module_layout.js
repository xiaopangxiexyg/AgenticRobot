"use strict";

// Presentation-only ordering for the right-side operator modules.
//
// The DOM order is a browser preference. This script emits no operator event,
// owns no robot/dataset state, and does not reconstruct lower-level state.
(function () {
  // v2 resets the preference once because the former combined
  // teleop/arm and collection/progress modules were split into independently
  // movable modules.
  const STORAGE_KEY = "dexteleop.operatorModuleOrder.v2";
  const COOKIE_KEY = "dexteleop_operator_module_order_v2";
  const host = document.querySelector("[data-operator-module-host]");
  if (!host) return;

  let draggedModule = null;

  function modules() {
    return Array.from(
      host.querySelectorAll(":scope > [data-operator-module-id]"),
    );
  }

  const defaultOrder = modules().map(
    (module) => String(module.dataset.operatorModuleId || ""),
  );

  function normalizedOrder(candidate) {
    const known = new Set(defaultOrder);
    const result = [];
    for (const rawId of Array.isArray(candidate) ? candidate : []) {
      const id = String(rawId || "").trim();
      if (known.has(id) && !result.includes(id)) result.push(id);
    }
    for (const id of defaultOrder) {
      if (!result.includes(id)) result.push(id);
    }
    return result;
  }

  function readCookie() {
    const prefix = `${COOKIE_KEY}=`;
    for (const part of String(document.cookie || "").split(";")) {
      const item = part.trim();
      if (item.startsWith(prefix)) {
        try {
          return decodeURIComponent(item.slice(prefix.length)).split(",");
        } catch (_error) {
          return [];
        }
      }
    }
    return [];
  }

  function readStoredOrder() {
    const cookieOrder = readCookie();
    if (cookieOrder.length) return normalizedOrder(cookieOrder);
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      return normalizedOrder(stored ? JSON.parse(stored) : []);
    } catch (_error) {
      return normalizedOrder([]);
    }
  }

  function applyOrder(order) {
    const byId = new Map(
      modules().map((module) => [
        String(module.dataset.operatorModuleId || ""),
        module,
      ]),
    );
    for (const id of normalizedOrder(order)) {
      const module = byId.get(id);
      if (module) host.appendChild(module);
    }
  }

  function currentOrder() {
    return modules().map(
      (module) => String(module.dataset.operatorModuleId || ""),
    );
  }

  function persistOrder() {
    const order = currentOrder();
    const encoded = encodeURIComponent(order.join(","));
    document.cookie = (
      `${COOKIE_KEY}=${encoded}; Max-Age=31536000; Path=/; SameSite=Lax`
    );
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(order));
    } catch (_error) {
      // A blocked localStorage does not disable same-session dragging.
    }
  }

  function finishDrag() {
    if (draggedModule) draggedModule.classList.remove("is-dragging");
    for (const module of modules()) {
      module.classList.remove("is-drag-target");
    }
    draggedModule = null;
  }

  function moveByKeyboard(module, direction) {
    const ordered = modules();
    const index = ordered.indexOf(module);
    const targetIndex = index + direction;
    if (index < 0 || targetIndex < 0 || targetIndex >= ordered.length) return;
    const target = ordered[targetIndex];
    if (direction < 0) {
      host.insertBefore(module, target);
    } else {
      host.insertBefore(module, target.nextSibling);
    }
    persistOrder();
    const handle = module.querySelector(".operator-module-drag-handle");
    if (handle) handle.focus();
  }

  function installModule(module) {
    const handle = module.querySelector(".operator-module-drag-handle");
    if (!handle) return;
    handle.addEventListener("dragstart", (event) => {
      draggedModule = module;
      module.classList.add("is-dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData(
          "text/plain",
          String(module.dataset.operatorModuleId || ""),
        );
      }
    });
    handle.addEventListener("dragend", () => {
      if (draggedModule) persistOrder();
      finishDrag();
    });
    handle.addEventListener("keydown", (event) => {
      if (event.key === "ArrowUp") {
        event.preventDefault();
        moveByKeyboard(module, -1);
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        moveByKeyboard(module, 1);
      }
    });
  }

  function installResetButton() {
    const hint = host.querySelector(":scope > .operator-layout-hint");
    if (!hint || document.getElementById("operatorModuleResetOrder")) return;
    const button = document.createElement("button");
    button.id = "operatorModuleResetOrder";
    button.type = "button";
    button.className = "operator-module-reset-order";
    button.textContent = "恢复默认顺序";
    button.addEventListener("click", () => {
      applyOrder(defaultOrder);
      persistOrder();
    });
    hint.appendChild(button);
  }

  function addStyle() {
    if (document.getElementById("operatorModuleLayoutStyle")) return;
    const style = document.createElement("style");
    style.id = "operatorModuleLayoutStyle";
    style.textContent = `
      .operator-layout-hint {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 10px;
        color: #69778b;
        font-size: 12px;
        line-height: 1.45;
      }
      .operator-module-reset-order {
        flex: 0 0 auto;
        padding: 6px 9px;
        background: #f3f6fa;
        font-size: 11px;
      }
      .operator-module {
        margin-top: 12px;
        border: 1px solid #d7dee8;
        border-radius: 8px;
        background: #fbfcfe;
        overflow: hidden;
        transition: opacity 120ms ease, border-color 120ms ease,
          box-shadow 120ms ease;
      }
      .operator-module:first-of-type {
        margin-top: 0;
      }
      .operator-module.is-dragging {
        opacity: 0.46;
      }
      .operator-module.is-drag-target {
        border-color: #147a74;
        box-shadow: 0 0 0 3px rgba(20, 122, 116, 0.13);
      }
      .operator-module-drag-handle {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        border-bottom: 1px solid #d7dee8;
        background: #eef4f8;
        color: #27364a;
        cursor: grab;
        font-size: 13px;
        font-weight: 850;
        user-select: none;
      }
      .operator-module-drag-handle:active {
        cursor: grabbing;
      }
      .operator-module-drag-handle:focus-visible {
        outline: 3px solid rgba(20, 122, 116, 0.24);
        outline-offset: -3px;
      }
      .operator-module-grip {
        color: #6a7f94;
        font-size: 18px;
        line-height: 1;
      }
      .operator-module-content {
        padding: 12px;
      }
      .operator-module-content > .control-title:first-child {
        margin-top: 0;
      }
      @media (max-width: 760px) {
        .operator-layout-hint {
          align-items: flex-start;
          flex-direction: column;
        }
      }
    `;
    document.head.appendChild(style);
  }

  addStyle();
  applyOrder(readStoredOrder());
  for (const module of modules()) installModule(module);
  installResetButton();

  host.addEventListener("dragover", (event) => {
    if (!draggedModule) return;
    const target = event.target.closest("[data-operator-module-id]");
    if (!target || target === draggedModule || target.parentNode !== host) return;
    event.preventDefault();
    for (const module of modules()) {
      module.classList.toggle("is-drag-target", module === target);
    }
    const rect = target.getBoundingClientRect();
    const insertBefore = event.clientY < rect.top + rect.height / 2;
    host.insertBefore(
      draggedModule,
      insertBefore ? target : target.nextSibling,
    );
  });
  host.addEventListener("drop", (event) => {
    if (!draggedModule) return;
    event.preventDefault();
    persistOrder();
    finishDrag();
  });
})();
