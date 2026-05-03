from __future__ import annotations

from typing import Any

try:
    from omegaconf.errors import ConfigAttributeError, ConfigKeyError

    _MISSING_CONFIG_ERRORS = (AttributeError, KeyError, ConfigAttributeError, ConfigKeyError)
except ImportError:
    _MISSING_CONFIG_ERRORS = (AttributeError, KeyError)


def safe_config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)

    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass

    try:
        return getattr(config, key)
    except _MISSING_CONFIG_ERRORS:
        return default


def get_positive_limit(config: Any, limit_name: str) -> int | None:
    limits = safe_config_get(config, "limits", None)
    if limits is None:
        return None

    value = safe_config_get(limits, limit_name, None)
    if value is None:
        return None

    normalized = int(value)
    return None if normalized <= 0 else normalized
