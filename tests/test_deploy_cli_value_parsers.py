from __future__ import annotations

import argparse

import pytest

from teleop_runtime import cli_value_parsers as shared_cli_value_parsers
from teleop_runtime.deploy import cli_value_parsers
from teleop_runtime.deploy import run_policy


_MIGRATED_NAMES = (
    "_non_negative_float",
    "_non_negative_int",
    "_parse_active_sides",
    "_parse_arm_max_angvel",
    "_parse_disconnected_gripper_sides",
    "_parse_dmtac_tactile_fields",
    "_parse_end_effector_sides",
    "_parse_float_tuple",
    "_parse_qpos6",
    "_parse_vector3",
    "_positive_float",
    "_positive_int",
    "_unit_interval_float",
)


def test_run_policy_preserves_migrated_parser_symbols() -> None:
    for name in _MIGRATED_NAMES:
        assert getattr(run_policy, name) is getattr(cli_value_parsers, name)


def test_generic_numeric_parsers_use_shared_implementations() -> None:
    for name in ("_positive_float", "_non_negative_int", "_positive_int"):
        assert getattr(cli_value_parsers, name) is getattr(
            shared_cli_value_parsers,
            name,
        )


@pytest.mark.parametrize(
    ("parser", "text", "expected"),
    (
        (cli_value_parsers._positive_float, "0.25", 0.25),
        (cli_value_parsers._unit_interval_float, "1", 1.0),
        (cli_value_parsers._non_negative_float, "0", 0.0),
        (cli_value_parsers._non_negative_int, "0", 0),
        (cli_value_parsers._positive_int, "2", 2),
    ),
)
def test_numeric_parsers_preserve_valid_values(parser, text, expected) -> None:
    assert parser(text) == expected


@pytest.mark.parametrize(
    ("parser", "text", "message"),
    (
        (cli_value_parsers._positive_float, "0", "value must be positive"),
        (
            cli_value_parsers._unit_interval_float,
            "1.1",
            "value must be in (0, 1]",
        ),
        (
            cli_value_parsers._non_negative_float,
            "-0.1",
            "value must be non-negative",
        ),
        (cli_value_parsers._non_negative_int, "-1", "value must be >= 0"),
        (cli_value_parsers._positive_int, "0", "value must be positive"),
    ),
)
def test_numeric_parsers_preserve_argument_errors(parser, text, message) -> None:
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parser(text)
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("both", ("left", "right")),
        ("right,left", ("left", "right")),
        (("right",), ("right",)),
        (["left", "left"], ("left",)),
    ),
)
def test_active_side_parser_preserves_order_and_aliases(value, expected) -> None:
    assert cli_value_parsers._parse_active_sides(value) == expected


@pytest.mark.parametrize("value", ("", "center", "both,left"))
def test_active_side_parser_preserves_rejections(value) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="active-sides must be one of: left, right, both",
    ):
        cli_value_parsers._parse_active_sides(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("", ()),
        ("none", ()),
        ("both", ("left", "right")),
        (["right", "left"], ("left", "right")),
    ),
)
def test_disconnected_gripper_side_parser_preserves_values(value, expected) -> None:
    assert cli_value_parsers._parse_disconnected_gripper_sides(value) == expected


def test_end_effector_side_parser_preserves_default_and_validation() -> None:
    parse = cli_value_parsers._parse_end_effector_sides

    assert parse("active", active_sides=("right",), hands_enabled=True) == ("right",)
    assert parse("active", active_sides=("right",), hands_enabled=False) == ()
    assert parse("none", active_sides=("left",), hands_enabled=False) == ()
    assert parse("right,left", active_sides=("left", "right"), hands_enabled=True) == (
        "left",
        "right",
    )

    with pytest.raises(
        ValueError,
        match="--end-effector-sides requires hands to be enabled",
    ):
        parse("left", active_sides=("left",), hands_enabled=False)
    with pytest.raises(
        ValueError,
        match="--end-effector-sides must be a subset of --active-sides, got right",
    ):
        parse("right", active_sides=("left",), hands_enabled=True)
    with pytest.raises(argparse.ArgumentTypeError, match="end-effector-sides"):
        parse("center", active_sides=("left", "right"), hands_enabled=True)


def test_tuple_vector_and_qpos_parsers_preserve_shapes_and_errors() -> None:
    assert cli_value_parsers._parse_float_tuple(
        "1, 2,3", length=3, name="values"
    ) == (1.0, 2.0, 3.0)
    assert cli_value_parsers._parse_vector3(["1,2", "3"]) == (1.0, 2.0, 3.0)
    assert cli_value_parsers._parse_qpos6("1,2,3,4,5,6") == (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    )

    with pytest.raises(argparse.ArgumentTypeError, match="values must contain 2"):
        cli_value_parsers._parse_float_tuple("1", length=2, name="values")
    with pytest.raises(argparse.ArgumentTypeError, match="invalid values: 1,x"):
        cli_value_parsers._parse_float_tuple("1,x", length=2, name="values")
    with pytest.raises(argparse.ArgumentTypeError, match="vector3 must contain 3"):
        cli_value_parsers._parse_vector3("1,2")
    with pytest.raises(argparse.ArgumentTypeError, match="invalid vector3: 1 x 3"):
        cli_value_parsers._parse_vector3(["1", "x", "3"])


def test_arm_max_angvel_parser_preserves_scalar_expansion_and_validation() -> None:
    parse = cli_value_parsers._parse_arm_max_angvel

    assert parse("0.5") == (0.5,) * 6
    assert parse("1,2,3,4,5,6") == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="arm max angvel must contain 1 or 6 floats",
    ):
        parse("")
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="arm max angvel must contain 1 or 6 positive floats",
    ):
        parse("1,2")
    with pytest.raises(argparse.ArgumentTypeError, match="arm max angvel must be positive"):
        parse("0")


def test_dmtac_tactile_field_parser_preserves_normalization_and_error_type() -> None:
    parse = cli_value_parsers._parse_dmtac_tactile_fields

    assert parse("raw, distributed-force, raw") == ("raw", "distributed_force")
    with pytest.raises(argparse.ArgumentTypeError, match="Unsupported DMTAC tactile field"):
        parse("unsupported")
