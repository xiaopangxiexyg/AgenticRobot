from __future__ import annotations

import ast
import inspect
from importlib import resources

from control_station.web import calibration_web_ui


def test_calibration_web_ui_exports_packaged_asset_contents() -> None:
    package_root = resources.files("control_station.web")

    html = package_root.joinpath("static/calibration_web_ui.html").read_text(
        encoding="utf-8"
    )
    fallback = package_root.joinpath("static/calibration_web_ui_fallback.js").read_text(
        encoding="utf-8"
    )

    assert calibration_web_ui._HTML_PAGE == html
    assert calibration_web_ui._FALLBACK_SCRIPT == fallback
    assert html.startswith("<!doctype html>")
    assert fallback.startswith("\n(function () {")


def test_calibration_web_ui_source_loads_assets_instead_of_embedding_them() -> None:
    source = inspect.getsource(calibration_web_ui)
    module = ast.parse(source)
    assignments = {
        target.id: node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"_HTML_PAGE", "_FALLBACK_SCRIPT"}
    }

    assert set(assignments) == {"_HTML_PAGE", "_FALLBACK_SCRIPT"}
    assert all(isinstance(value, ast.Call) for value in assignments.values())
    assert all(
        isinstance(value.func, ast.Name) and value.func.id == "_load_web_asset"
        for value in assignments.values()
        if isinstance(value, ast.Call)
    )
