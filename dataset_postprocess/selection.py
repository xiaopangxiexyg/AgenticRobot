from __future__ import annotations

from collections.abc import Iterable


def parse_episode_selection(selection: str | Iterable[int]) -> tuple[int, ...]:
    """Parse episode selections such as ``"1,3,5-8"`` into sorted indices."""
    if isinstance(selection, str):
        indices: set[int] = set()
        text = selection.strip()
        if not text:
            return ()
        for token in text.split(","):
            item = token.strip()
            if not item:
                continue
            if "-" in item:
                left, right = (part.strip() for part in item.split("-", 1))
                start = int(left)
                end = int(right)
                if start > end:
                    raise ValueError(f"Invalid descending episode range: {item!r}.")
                indices.update(range(start, end + 1))
            else:
                indices.add(int(item))
    else:
        indices = {int(index) for index in selection}
    if any(index < 0 for index in indices):
        raise ValueError("Episode indices must be non-negative.")
    return tuple(sorted(indices))
