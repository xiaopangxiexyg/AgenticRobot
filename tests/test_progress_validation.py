"""Characterize shared Progress primitives and their domain-specific errors."""

from __future__ import annotations

import importlib
import math

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.contract]

_MODULES = {
    "progress_annotations": ("ProgressAnnotationContractError", ("_finite", "_confidence", "_nonnegative_int")),
    "progress_detail_annotations": ("DetailStageAnnotationContractError", ("_finite",)),
    "progress_facts": ("ProgressFactContractError", ("_finite", "_positive_int")),
    "progress_failure_triggers": ("FailureTriggerContractError", ("_finite", "_confidence", "_nonnegative_int")),
    "progress_profile": ("ProgressProfileContractError", ("_positive_int",)),
    "progress_recoverable_failures": ("RecoverableFailureContractError", ("_finite", "_nonnegative_int")),
    "progress_refinement": ("ProgressRefinementContractError", ("_positive_int",)),
    "progress_skill_projection": ("ProgressSkillProjectionContractError", ()),
    "progress_stage_annotations": ("ProgressStageAnnotationContractError", ("_finite", "_confidence", "_positive_int")),
    "progress_stages": ("ProgressStageContractError", ("_positive_int",)),
    "retrospective_campaign": ("RetrospectiveCampaignContractError", ("_finite",)),
    "retrospective_progress": ("RetrospectiveProgressContractError", ("_finite", "_non_negative_int")),
}

_NONNEGATIVE_CASES = (
    ("zero", 0, 0, None),
    ("positive", 3, 3, None),
    ("negative", -1, None, "probe must be a non-negative integer."),
    ("bool", False, None, "probe must be a non-negative integer."),
    ("float", 0.0, None, "probe must be a non-negative integer."),
)
_CASES = {
    "_required": (
        ("text", "value", "value", None),
        ("trim", " \t值\n ", "值", None),
        ("empty", "", None, "probe must be a non-empty string."),
        ("blank", " \t\n", None, "probe must be a non-empty string."),
        ("null", None, None, "probe must be a non-empty string."),
        ("number", 5, None, "probe must be a non-empty string."),
        ("bytes", b"value", None, "probe must be a non-empty string."),
    ),
    "_sha256": (
        ("lower", "a" * 64, "a" * 64, None),
        ("normalize", " \t" + "ABCDEF01" * 8 + "\n", "abcdef01" * 8, None),
        ("short", "a" * 63, None, "probe must be a lowercase SHA-256 digest."),
        ("long", "a" * 65, None, "probe must be a lowercase SHA-256 digest."),
        ("nonhex", "g" * 64, None, "probe must be a lowercase SHA-256 digest."),
        ("blank", " ", None, "probe must be a non-empty string."),
        ("null", None, None, "probe must be a non-empty string."),
    ),
    "_finite": (
        ("integer", 4, 4.0, None),
        ("zero", 0, 0.0, None),
        ("negative_zero", -0.0, -0.0, None),
        ("negative", -2.5, -2.5, None),
        ("bool", True, None, "probe must be numeric."),
        ("text", "1.0", None, "probe must be numeric."),
        ("null", None, None, "probe must be numeric."),
        ("nan", float("nan"), None, "probe must be finite."),
        ("infinity", float("inf"), None, "probe must be finite."),
        ("negative_infinity", -float("inf"), None, "probe must be finite."),
    ),
    "_positive_int": (
        ("one", 1, 1, None),
        ("positive", 9, 9, None),
        ("zero", 0, None, "probe must be a positive integer."),
        ("negative", -1, None, "probe must be a positive integer."),
        ("bool", True, None, "probe must be a positive integer."),
        ("float", 1.0, None, "probe must be a positive integer."),
    ),
    "_nonnegative_int": _NONNEGATIVE_CASES,
    "_non_negative_int": _NONNEGATIVE_CASES,
    "_confidence": (
        ("zero", 0, 0.0, None),
        ("negative_zero", -0.0, -0.0, None),
        ("one", 1, 1.0, None),
        ("fraction", 0.25, 0.25, None),
        ("negative", -0.01, None, "confidence must lie in [0, 1]."),
        ("large", 1.01, None, "confidence must lie in [0, 1]."),
        ("nan", float("nan"), None, "confidence must be finite."),
        ("bool", False, None, "confidence must be numeric."),
        ("text", "0.5", None, "confidence must be numeric."),
    ),
}


@pytest.mark.parametrize(
    "module_name,error_name,helper_name,value,expected,message",
    [
        pytest.param(
            module_name, error_name, helper_name, value, expected, message,
            id=f"{module_name}-{helper_name}-{label}",
        )
        for module_name, (error_name, extra_helpers) in _MODULES.items()
        for helper_name in ("_required", "_sha256", *extra_helpers)
        for label, value, expected, message in _CASES[helper_name]
    ],
)
def test_progress_primitive_contract(
    module_name, error_name, helper_name, value, expected, message
):
    module = importlib.import_module(f"dataset_contracts.{module_name}")
    validate = getattr(module, helper_name)
    kwargs = {} if helper_name == "_confidence" else {"name": "probe"}
    if message is None:
        result = validate(value, **kwargs)
        assert result == expected
        assert type(result) is type(expected)
        if isinstance(expected, float):
            assert math.copysign(1.0, result) == math.copysign(1.0, expected)
    else:
        error_type = getattr(module, error_name)
        with pytest.raises(error_type) as caught:
            validate(value, **kwargs)
        assert type(caught.value) is error_type
        assert str(caught.value) == message


@pytest.mark.parametrize(
    "module_name",
    [name for name, (_, helpers) in _MODULES.items() if "_finite" in helpers],
)
def test_progress_float_conversion_overflow_is_not_reclassified(module_name):
    module = importlib.import_module(f"dataset_contracts.{module_name}")
    with pytest.raises(OverflowError):
        module._finite(10**400, name="probe")


@pytest.mark.parametrize("value,word", [(-1, "non-negative"), (True, "non-negative"), (0, "positive")])
def test_retrospective_positive_integer_keeps_its_distinct_failure_order(value, word):
    module = importlib.import_module("dataset_contracts.retrospective_progress")
    with pytest.raises(module.RetrospectiveProgressContractError) as caught:
        module._positive_int(value, name="probe")
    assert str(caught.value) == f"probe must be a {word} integer."
