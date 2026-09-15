from __future__ import annotations

import argparse
import ast
import inspect

import pytest

from teleop_runtime.deploy import deploy_cli_parser, run_policy


def _fingerprint(register) -> tuple[int, int, tuple[str, ...], tuple[tuple[str, ...], ...]]:
    parser = argparse.ArgumentParser(add_help=False)
    register(parser)
    return (
        len(parser._actions),
        sum(len(action.option_strings) for action in parser._actions),
        tuple(action.dest for action in parser._actions),
        tuple(tuple(action.option_strings) for action in parser._actions),
    )


def test_deployment_parser_preserves_compatibility_option_surface() -> None:
    extracted = _fingerprint(deploy_cli_parser.add_args)
    compatibility = _fingerprint(run_policy.add_args)

    assert extracted == compatibility
    assert extracted[0:2] == (276, 306)


def test_rtc_preflight_deadline_is_independent_from_executable_ttl() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    deploy_cli_parser.add_args(parser)
    defaults = parser.parse_args([])
    assert defaults.rtc_active_preflight_timeout_sec == 60.0
    assert defaults.rtc_request_validity_steps == 10
    changed = parser.parse_args(["--rtc-active-preflight-timeout-sec", "90"])
    assert changed.rtc_active_preflight_timeout_sec == 90.0
    assert changed.rtc_request_validity_steps == 10
    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--rtc-active-preflight-timeout-sec", value])


def test_deployment_parser_parses_representative_typed_options() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    deploy_cli_parser.add_args(parser)

    args = parser.parse_args(
        [
            "--policy",
            "hold",
            "--active-sides",
            "left",
            "--arm-max-angvel",
            "1.0,2.0,3.0,4.0,5.0,6.0",
            "--camera-mode",
            "none",
            "--position-reference-hz",
            "60",
        ]
    )

    assert args.policy_spec == "hold"
    assert args.active_sides == ["left"]
    assert args.arm_max_angvel == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert args.camera_mode == "none"
    assert args.position_reference_hz == 60.0


def test_run_policy_add_args_is_only_a_compatibility_wrapper() -> None:
    source = inspect.getsource(run_policy.add_args)

    assert "_register_deployment_cli_args(parser)" in source
    assert source.count("add_argument") == 0


def test_deployment_parser_does_not_import_run_policy() -> None:
    parser_tree = ast.parse(inspect.getsource(deploy_cli_parser))

    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name.endswith("run_policy") for alias in node.names)
        for node in ast.walk(parser_tree)
    )
    assert not any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").endswith("run_policy")
        for node in ast.walk(parser_tree)
    )


def test_deployment_parser_has_no_unused_import_bindings() -> None:
    parser_tree = ast.parse(inspect.getsource(deploy_cli_parser))
    imported_bindings: set[str] = set()
    for node in parser_tree.body:
        if isinstance(node, ast.Import):
            imported_bindings.update(
                alias.asname or alias.name.split(".", 1)[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imported_bindings.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name != "*"
            )

    loaded_bindings = {
        node.id
        for node in ast.walk(parser_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    assert sorted(imported_bindings - loaded_bindings) == []
