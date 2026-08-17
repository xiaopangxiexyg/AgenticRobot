#!/usr/bin/env python3
"""Validate the repository-local DexTeleop documentation graph."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


_INLINE_LINK_RE = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(?P<target><[^>\n]+>|[^)\s]+)(?:\s+[^)]*)?\)"
)
_REFERENCE_LINK_RE = re.compile(
    r"^\s*\[[^\]]+\]:\s*(?P<target><[^>\n]+>|\S+)"
)
_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")


@dataclass(frozen=True, order=True)
class DocIssue:
    """One deterministic documentation navigation violation."""

    code: str
    source: str
    line: int
    detail: str

    def format(self) -> str:
        location = f"{self.source}:{self.line}" if self.line else self.source
        return f"{location}: [{self.code}] {self.detail}"


@dataclass(frozen=True)
class DocAudit:
    """Summary of one documentation navigation audit."""

    docs_files: int
    markdown_files: int
    issues: tuple[DocIssue, ...]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _iter_markdown_targets(path: Path) -> tuple[tuple[int, str], ...]:
    targets: list[tuple[int, str]] = []
    active_fence: str | None = None

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fence_match = _FENCE_RE.match(line)
        if fence_match is not None:
            fence = fence_match.group("fence")
            if active_fence is None:
                active_fence = fence[0]
            elif active_fence == fence[0]:
                active_fence = None
            continue
        if active_fence is not None:
            continue

        targets.extend(
            (line_number, match.group("target"))
            for match in _INLINE_LINK_RE.finditer(line)
        )
        reference_match = _REFERENCE_LINK_RE.match(line)
        if reference_match is not None:
            targets.append((line_number, reference_match.group("target")))

    return tuple(targets)


def _local_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if not target or target.startswith("#"):
        return None

    parsed = urlsplit(target)
    if parsed.scheme or target.startswith("//"):
        return None
    if not parsed.path:
        return None
    return unquote(parsed.path)


def _resolve_link(source: Path, target_text: str) -> Path:
    return (source.parent / target_text).resolve()


def _document_target(path: Path) -> Path:
    if path.is_dir() and (path / "README.md").is_file():
        return path / "README.md"
    return path


def _reachable(
    entry: Path,
    *,
    edges: dict[Path, set[Path]],
    allowed: set[Path],
) -> set[Path]:
    seen: set[Path] = set()
    queue: deque[Path] = deque([entry])
    while queue:
        current = queue.popleft()
        if current in seen or current not in allowed:
            continue
        seen.add(current)
        queue.extend(edges.get(current, set()) - seen)
    return seen


def audit_docs(repo_root: Path) -> DocAudit:
    """Audit local docs links, top-level reachability, and evidence indexes."""

    repo_root = repo_root.resolve()
    docs_root = repo_root / "docs"
    entry = docs_root / "README.md"
    if not entry.is_file():
        issue = DocIssue(
            code="missing_docs_entry",
            source="docs/README.md",
            line=0,
            detail="canonical documentation entry does not exist",
        )
        return DocAudit(docs_files=0, markdown_files=0, issues=(issue,))

    docs_files = {
        path.resolve()
        for path in docs_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    markdown_files = {path for path in docs_files if path.suffix == ".md"}
    edges: dict[Path, set[Path]] = defaultdict(set)
    issues: list[DocIssue] = []

    for source in sorted(markdown_files):
        source_display = source.relative_to(repo_root).as_posix()
        for line_number, raw_target in _iter_markdown_targets(source):
            target_text = _local_target(raw_target)
            if target_text is None:
                continue
            target_path = Path(target_text)
            if target_path.is_absolute():
                issues.append(
                    DocIssue(
                        code="absolute_local_link",
                        source=source_display,
                        line=line_number,
                        detail=f"local link must be repository-relative: {raw_target}",
                    )
                )
                continue

            resolved = _resolve_link(source, target_text)
            if not _is_relative_to(resolved, repo_root):
                issues.append(
                    DocIssue(
                        code="local_link_outside_repository",
                        source=source_display,
                        line=line_number,
                        detail=f"local link escapes the repository: {raw_target}",
                    )
                )
                continue
            if _is_relative_to(resolved, repo_root / "todos"):
                issues.append(
                    DocIssue(
                        code="tracked_docs_link_to_todos",
                        source=source_display,
                        line=line_number,
                        detail=f"tracked documentation must not link local todos: {raw_target}",
                    )
                )
            if not resolved.exists():
                issues.append(
                    DocIssue(
                        code="broken_local_link",
                        source=source_display,
                        line=line_number,
                        detail=f"target does not exist: {raw_target}",
                    )
                )
                continue

            document_target = _document_target(resolved).resolve()
            if document_target in docs_files:
                edges[source].add(document_target)

    reachable_from_entry = _reachable(
        entry.resolve(), edges=edges, allowed=docs_files
    )
    for path in sorted(docs_files - reachable_from_entry):
        issues.append(
            DocIssue(
                code="unreachable_docs_file",
                source=path.relative_to(repo_root).as_posix(),
                line=0,
                detail="not reachable from docs/README.md through local Markdown links",
            )
        )

    for evidence_index in sorted(docs_root.glob("**/evidence/README.md")):
        evidence_root = evidence_index.parent.resolve()
        evidence_files = {
            path.resolve()
            for path in evidence_root.rglob("*")
            if path.is_file() and path.resolve() != evidence_index.resolve()
        }
        allowed = evidence_files | {evidence_index.resolve()}
        indexed = _reachable(evidence_index.resolve(), edges=edges, allowed=allowed)
        for path in sorted(evidence_files - indexed):
            issues.append(
                DocIssue(
                    code="unindexed_evidence",
                    source=path.relative_to(repo_root).as_posix(),
                    line=0,
                    detail=(
                        "not reachable from "
                        f"{evidence_index.relative_to(repo_root).as_posix()}"
                    ),
                )
            )

    return DocAudit(
        docs_files=len(docs_files),
        markdown_files=len(markdown_files),
        issues=tuple(sorted(set(issues))),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate repository-local links and navigation under docs/."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of scripts/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    audit = audit_docs(args.repo_root)
    if audit.issues:
        for issue in audit.issues:
            print(issue.format())
        print(f"documentation navigation failed: {len(audit.issues)} issue(s)")
        return 1

    print(
        "documentation navigation passed: "
        f"{audit.docs_files} file(s), {audit.markdown_files} Markdown file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
