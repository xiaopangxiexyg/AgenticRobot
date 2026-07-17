"""RM65 robot-model resources shared by simulation and runtime planning."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RM65_ASSET_DIR = REPO_ROOT / "assets" / "rm65_inspire_tac"


def asset_path(*parts: str | Path) -> Path:
    """Return a path below the RM65 motion-model asset directory."""

    return RM65_ASSET_DIR.joinpath(*parts)


def require_asset_path(*parts: str | Path) -> Path:
    """Return an existing RM65 motion-model asset or raise a clear error."""

    path = asset_path(*parts)
    if not path.exists():
        raise FileNotFoundError(f"RM65 asset not found: {path}")
    return path


__all__ = ["REPO_ROOT", "RM65_ASSET_DIR", "asset_path", "require_asset_path"]
