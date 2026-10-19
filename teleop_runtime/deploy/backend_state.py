"""Read a backend snapshot with only its explicitly supported options."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any


def read_backend_state(
    method: Callable[..., Any],
    *,
    optional_arguments: Mapping[str, Any],
) -> Any:
    """Select supported keywords before the single backend invocation.

    Legacy backends keep their no-argument call. A TypeError raised by the
    backend itself must propagate, not cause a second read or a diagnostic
    fallback that changes the snapshot being checked.
    """

    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method()
    accepts_arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported_arguments = {
        name: value
        for name, value in optional_arguments.items()
        if accepts_arbitrary_keywords
        or (
            name in parameters
            and parameters[name].kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
    }
    return method(**supported_arguments)
