(function installRuleWritingSelection(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.RuleWritingSelection = api;
}(typeof globalThis === "object" ? globalThis : window, () => {
  "use strict";

  const directionLabels = {up: "向上", right: "向右", down: "向下", left: "向左"};

  function describeWritingVariant(variant = {}, configuration = {}, choices = []) {
    const id = String(variant.selected_trajectory_id || "");
    if (!id) {
      const configured = String(configuration.trajectory_id || "未指定");
      return `实际书写待确定（配置默认 ${configured}）`;
    }
    const choice = choices.find((item) => item && item.value === id);
    const digit = /^digit_([1-9])_v\d+$/.exec(id);
    const latin = /^latin_([a-z])_v\d+$/.exec(id);
    const label = choice && choice.label || (digit
      ? `数字 ${digit[1]}`
      : latin ? `字母 ${latin[1].toUpperCase()}` : id);
    let detail = `实际书写：${label}（${id}）`;
    const direction = String(variant.selected_direction || "");
    if (direction) detail += ` · 方向 ${directionLabels[direction] || direction}`;
    const deviation = Number(variant.selected_direction_deviation_deg);
    if (Number.isFinite(deviation) && deviation !== 0) {
      detail += `（偏角 ${deviation > 0 ? "+" : ""}${deviation.toFixed(1)}°）`;
    }
    const size = Number(variant.selected_size_m);
    if (Number.isFinite(size) && size > 0) detail += ` · 字高 ${size.toFixed(3)}m`;
    return detail;
  }

  return {describeWritingVariant};
}));
