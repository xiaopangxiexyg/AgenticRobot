from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
REPO_ROOT = PROJECT_ROOT
RM65_INSPIRE_TAC_ASSET_DIR = PROJECT_ROOT / "assets" / "rm65_inspire_tac"


def asset_path(*parts: str) -> Path:
    return RM65_INSPIRE_TAC_ASSET_DIR.joinpath(*parts)


def require_asset_path(*parts: str) -> Path:
    path = asset_path(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Asset not found: {path}")
    return path
