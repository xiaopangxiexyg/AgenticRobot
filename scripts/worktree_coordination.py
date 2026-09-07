"""Shared branch-description format for standalone Git worktree coordination."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

ACTIVE_STATES = {
    "active",
    "ready-for-integration",
    "integrating",
    "validated-for-mainline",
}
ALLOWED_STATES = ACTIVE_STATES | {"blocked", "superseded"}
IMMUTABLE_STATES = {"ready-for-integration", "validated-for-mainline"}
REQUIRED_FIELDS = {
    "owner",
    "state",
    "base",
    "tip",
    "write-scope",
    "semantic-scope",
    "excludes",
    "depends-on",
    "validation",
}
LIST_FIELDS = {
    "input",
    "write-scope",
    "semantic-scope",
    "excludes",
    "depends-on",
    "validation",
}
KNOWN_FIELDS = REQUIRED_FIELDS | {"candidate", "input", "target-ref"}
LEGACY_KEYS = {
    "write_scope": "write-scope",
    "semantic_scope": "semantic-scope",
    "dependencies": "depends-on",
}


def parse_description(description: str) -> tuple[dict[str, list[str]], list[str]]:
    """Read canonical records and existing claim_worktree semicolon records.

    Legacy active records did not pin a tip. Reading them as pending permits
    inspection, but never makes them immutable or ready for integration.
    """
    legacy = description.lstrip().startswith("owner=")
    records = description.split(";") if legacy else description.splitlines()
    fields: dict[str, list[str]] = defaultdict(list)
    errors: list[str] = []
    for number, raw in enumerate(records, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=" if legacy else ":")
        key, value = key.strip().lower(), value.strip()
        if legacy:
            key = LEGACY_KEYS.get(key, key)
        if not separator or not key or not value:
            errors.append(f"description record {number} has an empty key or value")
        elif key not in KNOWN_FIELDS:
            errors.append(f"description record {number} has unknown field '{key}'")
        elif key not in LIST_FIELDS and fields[key]:
            errors.append(f"description field '{key}' must appear once")
        else:
            fields[key].append(value)
    if legacy and "tip" not in fields and "candidate" not in fields:
        fields["tip"] = ["pending"]
    missing = REQUIRED_FIELDS - fields.keys()
    if "candidate" in fields:
        missing.discard("tip")
    if missing:
        errors.append("missing description fields: " + ", ".join(sorted(missing)))
    state = fields.get("state", [None])[0]
    if state is not None and state not in ALLOWED_STATES:
        errors.append(f"unknown state '{state}'")
    if (
        "tip" in fields
        and "candidate" in fields
        and fields["tip"] != fields["candidate"]
    ):
        errors.append("tip and legacy candidate disagree")
    return dict(fields), errors


def serialize_description(fields: Mapping[str, str | Sequence[str]]) -> str:
    """Write one canonical key/value per line; repeat list-valued fields."""
    lines: list[str] = []
    for key, values in fields.items():
        for value in [values] if isinstance(values, str) else values:
            if not value.strip() or "\n" in value or "\r" in value:
                raise ValueError(
                    f"description field {key!r} must be nonempty and single-line"
                )
            lines.append(f"{key}: {value.strip()}")
    description = "\n".join(lines)
    _, errors = parse_description(description)
    if errors:
        raise ValueError("; ".join(errors))
    return description
