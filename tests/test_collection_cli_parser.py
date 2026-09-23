from __future__ import annotations

import argparse
import ast
import inspect
import sys
from pathlib import Path

import pytest

from teleop_runtime import cli_value_parsers, dmtac_tactile_cli
from teleop_runtime.collection import collect_cli_parser, run_collect
from teleop_runtime.collection.collect_cli_parser import (
    build_collection_parser,
    normalize_qpos_option_argv,
)
from teleop_runtime.real_teleop import cli_value_parsers as real_cli_value_parsers


def test_collection_and_real_teleop_share_identical_value_parsers() -> None:
    for name in (
        "_alpha_float",
        "_non_negative_float",
        "_non_negative_int",
        "_parse_joint_order",
    ):
        assert getattr(collect_cli_parser, name) is getattr(cli_value_parsers, name)
    for name in (
        "_non_negative_float",
        "_dmtac_worker_cpu_affinity",
        "_dmtac_worker_nice",
    ):
        owner = cli_value_parsers if name == "_non_negative_float" else dmtac_tactile_cli
        assert getattr(real_cli_value_parsers, name) is getattr(owner, name)
    assert (
        collect_cli_parser._dmtac_worker_cpu_affinity
        is dmtac_tactile_cli._dmtac_worker_cpu_affinity
    )
    assert collect_cli_parser._dmtac_worker_nice is dmtac_tactile_cli._dmtac_worker_nice

    assert collect_cli_parser._dmtac_worker_cpu_affinity("2, 4 2") == (2, 4)
    assert collect_cli_parser._parse_joint_order("5,4,3,2,1,0") == (5, 4, 3, 2, 1, 0)
    with pytest.raises(argparse.ArgumentTypeError, match="value must be >= 0"):
        collect_cli_parser._non_negative_float("-0.1")
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="CPU affinity indices must be >= 0",
    ):
        real_cli_value_parsers._dmtac_worker_cpu_affinity("-1")


def test_collection_parser_builds_full_compatibility_option_surface() -> None:
    parser = build_collection_parser()
    assert len(parser._actions) == 214
    assert sum(len(action.option_strings) for action in parser._actions) == 235

    args = parser.parse_args(
        normalize_qpos_option_argv(
            [
                "--active-sides",
                "right",
                "--return-right-qpos",
                "-1,-2,-3,-4,-5,-6",
                "--replay-group",
                "a:0-3",
                "--dmtac-tactile-worker-cpus",
                "2,4,2",
                "--hand-joint-order",
                "5,4,3,2,1,0",
            ]
        )
    )

    assert args.active_sides == ("right",)
    assert args.return_right_qpos == (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0)
    assert args.replay_group == ["a:0-3"]
    assert args.dmtac_tactile_worker_cpus == (2, 4)
    assert args.hand_joint_order == (5, 4, 3, 2, 1, 0)


def test_collection_parser_factory_returns_fresh_mutable_defaults() -> None:
    first = build_collection_parser().parse_args(["--replay-group", "a:0"])
    second = build_collection_parser().parse_args([])

    assert first.replay_group == ["a:0"]
    assert second.replay_group == []


def test_qpos_normalization_only_rewrites_supported_finite_vectors() -> None:
    assert normalize_qpos_option_argv(
        ["--return-left-qpos-deg", "-90,0,10,20,30,40", "--silent"]
    ) == [
        "--return-left-qpos-deg=-90,0,10,20,30,40",
        "--silent",
    ]
    assert normalize_qpos_option_argv(["--return-left-qpos", "1,2,3", "--silent"]) == [
        "--return-left-qpos",
        "1,2,3",
        "--silent",
    ]
    assert normalize_qpos_option_argv(["--return-left-qpos", "1,2,3,4,5,nan"]) == [
        "--return-left-qpos",
        "1,2,3,4,5,nan",
    ]


def test_run_collect_parse_args_keeps_zero_argument_compatibility_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert tuple(inspect.signature(run_collect.parse_args).parameters) == ()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--dataset-root",
            str(tmp_path),
            "--active-sides",
            "left",
            "--left-arm-ip",
            "192.0.2.10",
            "--no-q6-default-config",
            "--no-pose-frame-contract",
            "--camera-registry",
            "",
            "--no-camera",
            "--no-video",
            "--return-left-qpos-deg",
            "-90,0,10,20,30,40",
        ],
    )

    args = run_collect.parse_args()

    assert args.active_sides == ("left",)
    assert args.dataset_base_root == str(tmp_path)
    assert args.dataset_root.startswith(f"{tmp_path}/")
    assert args.return_left_qpos == pytest.approx(
        (
            -1.5707963267948966,
            0.0,
            0.17453292519943295,
            0.3490658503988659,
            0.5235987755982988,
            0.6981317007977318,
        )
    )
    assert args.write_video is False
    assert args.nominal_fps == args.hz


def test_run_collect_only_coordinates_parser_and_post_parse_validation() -> None:
    parse_args_source = inspect.getsource(run_collect.parse_args)
    parser_tree = ast.parse(inspect.getsource(collect_cli_parser))

    assert "build_collection_parser()" in parse_args_source
    assert ".add_argument(" not in parse_args_source
    assert run_collect.build_collection_parser is build_collection_parser
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name.endswith("run_collect") for alias in node.names)
        for node in ast.walk(parser_tree)
    )
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").endswith("run_collect")
        for node in ast.walk(parser_tree)
    )
