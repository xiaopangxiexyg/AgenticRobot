from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_docs_navigation.py"
SPEC = importlib.util.spec_from_file_location("check_docs_navigation", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_repo(root: Path) -> None:
    _write(
        root / "docs" / "README.md",
        "# Docs\n\n[Guide](./guide.md)\n\n[Evidence](./area/evidence/README.md)\n",
    )
    _write(root / "docs" / "guide.md", "# Guide\n")
    _write(
        root / "docs" / "area" / "evidence" / "README.md",
        "# Evidence\n\n[Receipt](./receipt.json)\n",
    )
    _write(root / "docs" / "area" / "evidence" / "receipt.json", "{}\n")


def _issue_codes(root: Path) -> set[str]:
    return {issue.code for issue in CHECKER.audit_docs(root).issues}


def test_current_repository_docs_navigation_is_valid() -> None:
    audit = CHECKER.audit_docs(REPO_ROOT)

    assert audit.docs_files > 0
    assert audit.markdown_files > 0
    assert audit.issues == ()


def test_minimal_docs_tree_is_valid(tmp_path: Path) -> None:
    _minimal_repo(tmp_path)

    assert CHECKER.audit_docs(tmp_path).issues == ()


def test_broken_local_link_is_reported(tmp_path: Path) -> None:
    _minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "guide.md",
        "# Guide\n\n[Missing](./missing.md)\n",
    )

    assert "broken_local_link" in _issue_codes(tmp_path)


def test_unreachable_docs_file_is_reported(tmp_path: Path) -> None:
    _minimal_repo(tmp_path)
    _write(tmp_path / "docs" / "orphan.md", "# Orphan\n")

    assert "unreachable_docs_file" in _issue_codes(tmp_path)


def test_unindexed_evidence_is_reported(tmp_path: Path) -> None:
    _minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "area" / "evidence" / "unindexed.json",
        "{}\n",
    )

    assert "unindexed_evidence" in _issue_codes(tmp_path)


def test_tracked_docs_must_not_link_root_todos(tmp_path: Path) -> None:
    _minimal_repo(tmp_path)
    _write(tmp_path / "todos" / "local-plan.md", "# Local\n")
    _write(
        tmp_path / "docs" / "guide.md",
        "# Guide\n\n[Local plan](../todos/local-plan.md)\n",
    )

    assert "tracked_docs_link_to_todos" in _issue_codes(tmp_path)


def test_local_links_must_stay_inside_repository(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _minimal_repo(repo_root)
    _write(tmp_path / "outside.md", "# Outside\n")
    _write(
        repo_root / "docs" / "guide.md",
        "# Guide\n\n[Outside](../../outside.md)\n",
    )

    assert "local_link_outside_repository" in _issue_codes(repo_root)


def test_absolute_local_links_are_reported(tmp_path: Path) -> None:
    _minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "guide.md",
        "# Guide\n\n[Machine path](/home/user/project/README.md)\n",
    )

    assert "absolute_local_link" in _issue_codes(tmp_path)


def test_external_links_and_fenced_examples_are_not_local_targets(
    tmp_path: Path,
) -> None:
    _minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "guide.md",
        """# Guide

[External](https://example.com/missing)

```markdown
[Example only](./not-a-real-link.md)
```
""",
    )

    assert CHECKER.audit_docs(tmp_path).issues == ()
